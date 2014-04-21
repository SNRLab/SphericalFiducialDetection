cmake_minimum_required(VERSION 2.8.9)

project(SphericalFiducialDetection)

#-----------------------------------------------------------------------------
set(EXTENSION_HOMEPAGE "http://www.slicer.org/slicerWiki/index.php/Documentation/4.3/Extensions/SphericalFiducialDetection")
set(EXTENSION_CATEGORY "Registration")
set(EXTENSION_CONTRIBUTORS "Laurent Chauvin (BWH), Junichi Tokuda (BWH)")
set(EXTENSION_DESCRIPTION "This module detects spherical fiducials in a volume and create a list of markups at the center of the detected fiducials")
set(EXTENSION_ICONURL "http://viewvc.slicer.org/viewvc.cgi/Slicer4/trunk/Extensions/Testing/SphericalFiducialDetection/SphericalFiducialDetection.png?revision=21745&view=co")
set(EXTENSION_SCREENSHOTURLS "http://wiki.slicer.org/slicerWiki/images/a/ab/Slicer-r19441-SphericalFiducialDetection-screenshot.png http://wiki.slicer.org/slicerWiki/images/1/1e/Slicer-r19441-SphericalFiducialDetection-screenshot-2.png")

#-----------------------------------------------------------------------------
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
add_subdirectory(SphericalFiducialDetection)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
