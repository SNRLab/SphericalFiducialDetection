cmake_minimum_required(VERSION 2.8.9)

project(PointillismRegistration)

#-----------------------------------------------------------------------------
set(EXTENSION_HOMEPAGE "http://www.slicer.org/slicerWiki/index.php/Documentation/4.3/Extensions/PointillismRegistration")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Laurent Chauvin (BWH), Junichi Tokuda (BWH)")
set(EXTENSION_DESCRIPTION "This is an example of extension bundling one CLI module")
set(EXTENSION_ICONURL "http://viewvc.slicer.org/viewvc.cgi/Slicer4/trunk/Extensions/Testing/PointillismRegistration/PointillismRegistration.png?revision=21745&view=co")
set(EXTENSION_SCREENSHOTURLS "http://wiki.slicer.org/slicerWiki/images/a/ab/Slicer-r19441-PointillismRegistration-screenshot.png http://wiki.slicer.org/slicerWiki/images/1/1e/Slicer-r19441-PointillismRegistration-screenshot-2.png")

#-----------------------------------------------------------------------------
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
add_subdirectory(PointillismRegistration)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
