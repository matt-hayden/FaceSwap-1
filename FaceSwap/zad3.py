#! /usr/bin/env python2
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('FaceSwap')
debug, info, warn = logger.debug, logger.info, logger.warn

import os, os.path

import cv2
import numpy as np


from drawing import *
import FaceRendering
import ImageProcessing
import models
import NonLinearLeastSquares
import utils


#the smaller this value gets the faster the detection will work
#if it is too small, the user's face might not be detected
maxImageSizeForDetection = 240


mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel()
projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])
debug( "mean3DShape is {}".format(len(mean3DShape)) )
debug( "blendshapes is {}".format(len(blendshapes)) )
debug( "mesh is {}".format(len(mesh)) )
debug( "idxs2D is {}".format(len(idxs2D)) )
debug( "idxs3D is {}".format(len(idxs3D)) )
debug( "projectionModel is {}".format(projectionModel) )


def swap_many(face_filenames, head_filenames, examine_keypoints=False, **kwargs):
	fst = []
	for f in face_filenames:
		ffilepart, _ = os.path.splitext(os.path.basename(f))
		i = cv2.imread(f)
		k = utils.getFaceKeypoints(i, maxImageSizeForDetection)
		if k and (len(k) == 1):
			t = utils.getFaceTextureCoords(k[0], mean3DShape, blendshapes, idxs2D, idxs3D)
			fst.append((f, i.shape, t))
			if examine_keypoints:
				for shape2D in k:
					drawPoints(i, shape2D.T)
				assert cv2.imwrite('detect-'+ffilepart+'.jpg', i)
		else:
			warn("Skipping {}: {} faces detected".format(f, len(k)))
	hss = []
	for h in head_filenames:
		hfilepart, _ = os.path.splitext(os.path.basename(h))
		i = cv2.imread(h)
		k = utils.getFaceKeypoints(i, maxImageSizeForDetection)
		if k and (len(k) == 1):
			hss.append((h, i.shape, k))
			if examine_keypoints:
				for shape2D in k:
					drawPoints(i, shape2D.T)
				assert cv2.imwrite('detect-'+hfilepart+'.jpg', i)
		else:
			warn("Skipping {}: {} faces detected".format(f, len(k)))
	info( "Permuting {} faces and {} heads (n={})".format(len(fst), len(hss), len(fst)*len(hss)) )
	for hfilename, hsize, shapes in hss:
		hfilepart, _ = os.path.splitext(os.path.basename(hfilename))
		backgroundImg = cv2.imread(hfilename)
		for ffilename, fsize, textureCoords in fst:
			ffilepart, _ = os.path.splitext(os.path.basename(ffilename))
			textureImg = cv2.imread(ffilename)
			masked, blended = swap_images(textureImg=textureImg,
										  backgroundImg=backgroundImg.copy(),
										  textureCoords=textureCoords,
										  shapes2D=shapes,
										  **kwargs)
			assert cv2.imwrite(hfilepart+'.'+ffilepart+'.jpg', blended)
			assert cv2.imwrite(hfilepart+'.'+ffilepart+'-masked.jpg', masked)


def swap_images(textureImg, backgroundImg, drawOverlay=False, lockedTranslation=False,
				textureCoords=None, renderer=None, shapes2D=None):
	assert textureImg is not None
	if textureCoords is None:
		textureKeypoints = utils.getFaceKeypoints(textureImg, maxImageSizeForDetection)
		textureCoords = utils.getFaceTextureCoords(textureKeypoints[0], mean3DShape, blendshapes, idxs2D, idxs3D)
	debug( "textureCoords is {}".format(len(textureCoords)) )

	assert backgroundImg is not None
	if renderer is None:
		renderer = FaceRendering.FaceRenderer(backgroundImg, textureImg, textureCoords, mesh)
	debug( "renderer is {}".format(renderer) )

	if shapes2D is None:
		shapes2D = utils.getFaceKeypoints(backgroundImg, maxImageSizeForDetection)
	debug( "shapes2D is {}".format(len(shapes2D)) )

	assert shapes2D
	for shape2D in shapes2D:
		#3D model parameter initialization
		modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

		#3D model parameter optimization
		args = ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D])
		modelParams = NonLinearLeastSquares.GaussNewton(modelParams,
                        projectionModel.residual,
						projectionModel.jacobian,
						args)

		#rendering the model to an image
		shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
		renderedImg = renderer.render(shape3D)

		#blending of the rendered face with the image
		mask = np.copy(renderedImg[:, :, 0])
		renderedImg = ImageProcessing.colorTransfer(backgroundImg, renderedImg, mask)
		backgroundImg = ImageProcessing.blendImages(renderedImg, backgroundImg, mask)
   

		#drawing of the mesh and keypoints
		if drawOverlay:
			drawPoints(backgroundImg, shape2D.T)
			drawProjectedShape(backgroundImg,
			  [mean3DShape, blendshapes],
			  projectionModel,
			  mesh,
			  modelParams,
			  lockedTranslation)
	return renderedImg, backgroundImg
#
from glob import glob
import sys
args = sys.argv[1:]

#swap_many(["../data/jolie.jpg"], ["../data/brad pitt.jpg"])
swap_many(glob('faces/*.*'), glob('heads/*.*'))
