#!/usr/bin/env python

import os
import numpy as np
import fnmatch
from osgeo import ogr,osr
import cameratransform as ct
from import_export_geotiff import ImportGeoTiff
from subprocess import Popen, PIPE
import sys
import time
import pylab as plt
import psutil

####Script to orthorectify DJI M350 or Mavic3 drone imagery
####N. Neckel 2024

def GetFileList(path,wildcard):
	filelist = []
	for file in os.listdir(path):
		if fnmatch.fnmatch(file, wildcard):
			filelist = np.append(filelist,file)
	return np.sort(filelist)

from cameratransform.parameter_set import ParameterSet, Parameter, ClassWithParameterSet, TYPE_EXTRINSIC1, TYPE_EXTRINSIC2
class SpatialOrientationYawPitchRollFaceDown(ct.SpatialOrientation):
    r"""
    The orientation in the yaw-pitch-roll system. The definition is based on
    `this <https://en.wikipedia.org//wiki/Aircraft_principal_axes>`_. It relates to the tilt-heading-roll system used
    by SpatialOrientation like this:

    .. math::
        R_{\mathrm{pitch}} &= 90+R_{\mathrm{tilt}} \\
        R_{\mathrm{yaw}} &= -R_{\mathrm{heading}} \\
        R_{\mathrm{roll}} &= -R_{\mathrm{roll}}

    """

    def __init__(self, elevation_m=None, yaw_deg=None, pitch_deg=None, roll_deg=None, cam_roll_deg=None, pos_x_m=None, pos_y_m=None):
        self.parameters = ParameterSet(
            # the extrinsic parameters if the camera will not be compared to other cameras or maps
            elevation_m=Parameter(elevation_m, default=30, range=(0, None), type=TYPE_EXTRINSIC1),
            # the elevation of the camera above sea level in m
            yaw_deg=Parameter(yaw_deg, default=85, range=(-90, 90), type=TYPE_EXTRINSIC1),  # the tilt angle of the camera in degrees
            roll_deg=Parameter(roll_deg, default=0, range=(-180, 180), type=TYPE_EXTRINSIC1),  # the roll angle of the camera in degrees

            # the extrinsic parameters if the camera will be compared to other cameras or maps
            pitch_deg=Parameter(pitch_deg, default=0, type=TYPE_EXTRINSIC2),  # the heading angle of the camera in degrees
            cam_roll_deg=Parameter(cam_roll_deg, default=0, range=(-180, 180), type=TYPE_EXTRINSIC1),  # the roll angle of the camera in degrees
            pos_x_m=Parameter(pos_x_m, default=0, type=TYPE_EXTRINSIC2),  # the x position of the camera in m
            pos_y_m=Parameter(pos_y_m, default=0, type=TYPE_EXTRINSIC2),  # the y position of the camera in m
        )
        for name in self.parameters.parameters:
            self.parameters.parameters[name].callback = self._initCameraMatrix
        self._initCameraMatrix()

    def __str__(self):
        string = ""
        string += "  position:\n"
        string += "    x:\t%f m\n    y:\t%f m\n    h:\t%f m\n" % (self.parameters.pos_x_m, self.parameters.pos_y_m, self.parameters.elevation_m)
        string += "  orientation:\n"
        string += "    yaw:\t\t%f°\n    pitch:\t\t%f°\n    roll:\t%f°\n" % (self.parameters.yaw_deg, self.parameters.pitch_deg, self.parameters.roll_deg)
        return string

    def _initCameraMatrix(self):
        if self.pitch_deg < -360 or self.pitch_deg > 360:  # pragma: no cover
            self.pitch_deg = self.pitch_deg % 360
        # convert the angle to radians
        tilt = np.deg2rad(self.parameters.pitch_deg)
        roll = -np.deg2rad(self.parameters.roll_deg)
        heading = np.deg2rad(self.parameters.yaw_deg)

        # get the translation matrix and rotate it
        self.t = np.array([self.parameters.pos_x_m, self.parameters.pos_y_m, self.parameters.elevation_m])

        # construct the rotation matrices for tilt, roll and heading
        roll0 = np.deg2rad(self.cam_roll_deg)
        self.R_roll0 = np.array([[np.cos(roll0), -np.sin(roll0), 0],
                                [np.sin(roll0), np.cos(roll0), 0],
                                [0, 0, 1]])
        
        self.R_roll = np.array([[+np.cos(roll), 0, np.sin(roll)],
                                [0, 1, 0],
                                [-np.sin(roll), 0, np.cos(roll)]])
        self.R_tilt = np.array([[1, 0, 0],
                                [0, np.cos(tilt), np.sin(tilt)],
                                [0, -np.sin(tilt), np.cos(tilt)]])
        self.R_head = np.array([[np.cos(heading), -np.sin(heading), 0],
                                [np.sin(heading), np.cos(heading), 0],
                                [0, 0, 1]])

        self.R = np.dot(np.dot(np.dot(self.R_roll0, self.R_roll), self.R_tilt), self.R_head)
        self.R_inv = np.linalg.inv(self.R)

def EstimateFootprintGDAL(img,lat,lon,alt,head,tilt,roll,JPGpath,EPSG,DEM_FILE):
	cam = ct.Camera(ct.RectilinearProjection(focallength_px=f,image=image_size), SpatialOrientationYawPitchRollFaceDown(elevation_m=alt, yaw_deg=head, pitch_deg=tilt, roll_deg=roll, cam_roll_deg=0))
	cam.setGPSpos(lat,lon,alt)
	coords = np.array([cam.gpsFromImage([0 , 0]), cam.gpsFromImage([image_size[0]-1 , 0]), cam.gpsFromImage([image_size[0]-1, image_size[1]-1]), cam.gpsFromImage([0 , image_size[1]-1]), cam.gpsFromImage([0 , 0])])
	x_coords = []
	y_coords = []
	ring = ogr.Geometry(ogr.wkbLinearRing)	
	for m in np.arange(len(coords)):
		ring.AddPoint(coords[m][1],coords[m][0])
		if m<4:
			psx,psy,psz = TransCoordsLatLonToPS(EPSG).TransformPoint(coords[m][1],coords[m][0]) ##needed for worldfile
			x_coords = np.append(x_coords,psx)
			y_coords = np.append(y_coords,psy)
	poly = ogr.Geometry(ogr.wkbPolygon)
	poly.AddGeometry(ring)
	if calc_worldfile == 1:
		m,n = image_size[0],image_size[1]
		fp = np.matrix([[1,m,m,1],[1,1,n,n]])
		newligne = [1,1,1,1]
		fp  = np.vstack([fp,newligne])
		tp = np.matrix([x_coords,y_coords])
 
		# solution = fp x inverse(tp)
		M = tp * fp.I

		A = M[:, 0][0]
		B = M[:, 1][0]
		C = M[:, 2][0]
		D = M[:, 0][1]
		E = M[:, 1][1]
		F = M[:, 2][1]

		jpw = open(JPGpath+img[:-4]+'.jgw','w')
		jpw.write(str(A[0,0])+'\n')
		jpw.write(str(D[0,0])+'\n')
		jpw.write(str(B[0,0])+'\n')
		jpw.write(str(E[0,0])+'\n')
		jpw.write(str(C[0,0])+'\n')
		jpw.write(str(F[0,0])+'\n')
		jpw.close()
	
	if calc_geotifs == 1 and os.path.exists(JPGpath+img):
		if not os.path.exists(dirname+'/geotifs'):
			os.mkdir(dirname+'/geotifs')
			
		im = plt.imread(JPGpath+img)
		plt.imsave(img[:-4]+'.png',np.ascontiguousarray(np.flipud(np.fliplr(im))),format='png')
		cmd = 'gdal_translate -gcp '+str(image_size[0])+' '+str(image_size[1])+' '+str(coords[0,1])+' '+str(coords[0,0])+' -gcp 0 '+str(image_size[1])+' '+str(coords[1,1])+' '+str(coords[1,0])+' -gcp 0 0 '+str(coords[2,1])+' '+str(coords[2,0])+' -gcp '+str(image_size[0])+' 0 '+str(coords[3,1])+' '+str(coords[3,0])+' '+img[:-4]+'.png tmp.tif'
		p = Popen(cmd, shell=True)
		p.wait()
		cmd = 'gdalwarp -r bilinear -ot Byte -s_srs EPSG:4326 -t_srs EPSG:'+str(EPSG)+' -overwrite tmp.tif '+dirname+'/geotifs/'+img[:-4]+'.tif'
		p = Popen(cmd, shell=True)
		p.wait()
		os.system('rm tmp.tif '+img[:-4]+'.png')
	return poly

def EstimateFootprintAMES(img,lat,lon,alt,head,tilt,roll,JPGpath,EPSG,DEM_FILE):
	cam = ct.Camera(ct.RectilinearProjection(focallength_px=f,image=image_size), SpatialOrientationYawPitchRollFaceDown(elevation_m=alt, yaw_deg=head, pitch_deg=tilt, roll_deg=roll, cam_roll_deg=0))
	cam.setGPSpos(lat,lon,alt)
	coords = np.array([cam.gpsFromImage([0 , 0]), cam.gpsFromImage([image_size[0]-1 , 0]), cam.gpsFromImage([image_size[0]-1, image_size[1]-1]), cam.gpsFromImage([0 , image_size[1]-1]), cam.gpsFromImage([0 , 0])])
	x_coords = []
	y_coords = []
	ring = ogr.Geometry(ogr.wkbLinearRing)
	for m in np.arange(len(coords)):
		ring.AddPoint(coords[m][1],coords[m][0])
	poly = ogr.Geometry(ogr.wkbPolygon)
	poly.AddGeometry(ring)
	coords = coords.astype(str)
	coordlist = coords[0,1]+' '+coords[0,0]+', '+coords[1,1]+' '+coords[1,0]+', '+coords[2,1]+' '+coords[2,0]+', '+coords[3,1]+' '+coords[3,0]
	print(coordlist)
	cmd = 'cam_gen --reference-dem {} --lon-lat-values "{}" {} -o {} --focal-length {} --pixel-pitch 1'.format(DEM_FILE,coordlist,JPGpath+img,JPGpath+img[:-4]+'.tsai',f)
	p = Popen(cmd, shell=True)
	p.wait()
	if calc_geotifs == 1 and os.path.exists(JPGpath+img):
		if not os.path.exists(dirname+'geotifs'):
			os.mkdir(dirname+'geotifs')
		cmd = 'mapproject {} {} {} {}'.format(DEM_FILE,JPGpath+img,JPGpath+img[:-4]+'.tsai',dirname+'geotifs/'+img[:-4]+'.tif')
		p = Popen(cmd, shell=True)
		p.wait()
	return poly

def TransCoordsLatLonToPS(EPSG):
	srs_in = osr.SpatialReference()
	srs_in.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
	srs_in.ImportFromEPSG(4326)
	srs_out = osr.SpatialReference()
	srs_out.ImportFromEPSG(EPSG)
	ct = osr.CoordinateTransformation(srs_in,srs_out)
	return ct

def TransCoordsPSToLatLon(EPSG):
	srs_in_back = osr.SpatialReference()
	srs_in_back.ImportFromEPSG(EPSG)
	srs_out_back = osr.SpatialReference()
	srs_out_back.ImportFromEPSG(4326)
	srs_out_back.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
	ct_back = osr.CoordinateTransformation(srs_in_back,srs_out_back)
	return ct_back

DEM_FILE_EUR = '/home/nneckel/DEMs/egm2008-1_BREMEN.tif'
DEM_EUR,DEMwidth,DEMheight,DEM_EUR_geotrans,DEMproj = ImportGeoTiff(DEM_FILE_EUR)
DEM_FILE_NORTH = '/home/nneckel/DEMs/AW3D30_Arctic_DTU21_1km.tif'
DEM_NORTH,DEMwidth,DEMheight,DEM_NORTH_geotrans,DEMproj = ImportGeoTiff(DEM_FILE_NORTH)

dirname = '/home/nneckel/ARC/PS144/UAV/direct_geo/'
JPGpath = dirname+'JPG4/'
calc_geotifs = 1
calc_worldfile = 1
filelist = GetFileList(JPGpath,'*JPG')
outfile = []

for i in np.arange(len(filelist)):
	tags = os.popen('exiftool {} -gpslatitude -gpslongitude -c "%.6f" -relativealtitude -flightyawdegree -flightpitchdegree -flightrolldegree -gimbalyawdegree -gimbalpitchdegree -gimbalrolldegree -createdate -focallength -absolutealtitude -exifimagewidth'.format(JPGpath+filelist[i])).readlines()
	print('Processing image {}'.format(filelist[i]))
	GPS_lat = float(tags[0].split(':')[1][1:-2])
	GPS_lon = float(tags[1].split(':')[1][1:-2])
	REL_alt = float(tags[2].split(':')[1].split(' ')[1])
	INS_yaw = float(tags[6].split(':')[1])
	FLIGHT_yaw = float(tags[3].split(':')[1])
	IMG_WIDTH = int(tags[12].split(':')[1])
	GPS_alt = float(tags[11].split(':')[1])
	INS_pitch = float(tags[7].split(':')[1])+90
	INS_roll = float(tags[8].split(':')[1])
	if float(GPS_lat) < 55:
		EPSG = 32632
		psx,psy,psz = TransCoordsLatLonToPS(EPSG).TransformPoint(float(GPS_lon),float(GPS_lat))
		xcoord = int(((psx - DEM_EUR_geotrans[0]) / DEM_EUR_geotrans[1]))
		ycoord = int(((psy - DEM_EUR_geotrans[3]) / DEM_EUR_geotrans[5]))
		refALT = DEM_EUR[ycoord,xcoord]
		DEM_FILE = DEM_FILE_EUR
	else:
		EPSG = 3413
		psx,psy,psz = TransCoordsLatLonToPS(EPSG).TransformPoint(float(GPS_lon),float(GPS_lat))
		xcoord = int(((psx - DEM_NORTH_geotrans[0]) / DEM_NORTH_geotrans[1]))
		ycoord = int(((psy - DEM_NORTH_geotrans[3]) / DEM_NORTH_geotrans[5]))
		refALT = DEM_NORTH[ycoord,xcoord]
		DEM_FILE = DEM_FILE_NORTH
	if IMG_WIDTH == 4056:
		print('Using H20T RGB Camera Model')
		image_size = (4056,3040)  #in px
		f_mm = 4.5 #in mm or 24 mm
		pixel_size = 0.00155 #in mm
		sensor_size = (image_size[0]*pixel_size,image_size[1]*pixel_size)
		f = f_mm*1/pixel_size
	if IMG_WIDTH == 640:
		print('Using H20T TIR Camera Model')
		image_size = (640,512)
		f_mm = 13.5
		pixel_size = 0.012
		sensor_size = (image_size[0]*pixel_size,image_size[1]*pixel_size)
		f = f_mm*1/pixel_size
	if IMG_WIDTH == 5280:
		print('Using Hasselblad L2D-20c Camera Model')
		image_size = (5280,3956)
		f_mm = 12.3
		pixel_size = 0.0033
		sensor_size = (image_size[0]*pixel_size,image_size[1]*pixel_size)
		f = f_mm*1/pixel_size
		GPS_alt = REL_alt+refALT+10 #This is a new altitude value composed out of barometric relative height + Mean Sea Level + Helideck altitude
		INS_yaw = FLIGHT_yaw
	start_time = time.time()
	poly = EstimateFootprintAMES(filelist[i],float(GPS_lat),float(GPS_lon),float(GPS_alt)-refALT,float(INS_yaw),float(INS_pitch),float(INS_roll),JPGpath,EPSG,DEM_FILE)
	outfile = np.append(outfile,np.array([filelist[i],str(GPS_lat),str(GPS_lon),str(GPS_alt),str(INS_yaw),str(INS_pitch),str(INS_roll),str(poly)]))
	print('CPU used: {}%'.format(psutil.cpu_percent()))
	print('memory % used:', psutil.virtual_memory()[2])
	print("--- %s seconds ---" % (time.time() - start_time))
outfile = outfile.reshape(int(len(outfile)/8),8)
np.savetxt('DJI.csv',outfile, fmt='%s', delimiter=';', newline='\n', header='image;lat;lon;alt;yaw;pitch;roll;WKT', footer='', comments='', encoding=None)
