# UMANS: Unified Microscopic Agent Navigation Simulator
# MIT License
# Copyright (C) 2018-2020  Inria Rennes Bretagne Atlantique - Rainbow - Julien Pettré
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject
# to the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
# Contact: crowd_group@inria.fr
# Website: https://project.inria.fr/crowdscience/
# See the file AUTHORS.md for a list of all contributors.

cmake_minimum_required( VERSION 3.10 )
project( UMANS )
set (CMAKE_CXX_STANDARD 17)

#set(EXECUTABLE_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

file( GLOB_RECURSE src_All src/*)
file( GLOB_RECURSE src_Engine src/Engine/* src/3rd-party/* )
file( GLOB_RECURSE src_Library src/Library/* )
file( GLOB_RECURSE src_ConsoleApplication src/ConsoleApplication/* )
file( GLOB_RECURSE src_GUI src/GUI/* )

# --- ensure that certain directories are recognized in "#include <...>" lines
include_directories( ./src/ ./src/Engine/ )

# === Include the GUI application (which requires Qt)?

option(UMANS_USE_GUI_APPLICATION "Include a GUI demo application (which relies on Qt)" ON)

# === Use multithreading?

option(UMANS_USE_MULTITHREADING "Allow the program to use multiple parallel threads" ON)
if (UMANS_USE_MULTITHREADING)
	if (MSVC)
	  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /openmp")
	else()
	  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif ()
endif (UMANS_USE_MULTITHREADING)

# === Linux compatibility compiler option

if (NOT MSVC)
	add_definitions(-O3 -lstdc++fs -fPIC -shared)
endif() 

# === Engine project: a static library that performs the simulation

add_library(Engine STATIC ${src_Engine})

# === Console application

if (WIN32)
	add_executable(UMANS-ConsoleApplication-Windows ${src_ConsoleApplication})
	target_link_libraries(UMANS-ConsoleApplication-Windows Engine)
else()
	add_executable(UMANS-ConsoleApplication-Linux  ${src_ConsoleApplication})
	target_link_libraries(UMANS-ConsoleApplication-Linux Engine)
endif()

# === Dynamic library (e.g. a DLL on Windows) to include into other programs

add_library(UMANS-Library SHARED ${src_Library})
target_link_libraries(UMANS-Library Engine)

# === Qt demo project (if enabled)

if (UMANS_USE_GUI_APPLICATION)

	# necessary: gives the path to GUI src files to qt autogen files
	include_directories( ./src/GUI/ )

	# set autogen variables for qt
	set(CMAKE_INCLUDE_CURRENT_DIR ON)
	set(CMAKE_AUTOUIC ON)
	set(CMAKE_AUTOMOC ON)
	set(CMAKE_AUTORCC ON)
	
	# find relevant packages
	find_package(Qt5Core REQUIRED)
	find_package(Qt5Widgets REQUIRED)
	find_package(OpenGL REQUIRED)

	# --- include the Qt resources, so that the application icon gets shown
	set(UI_RESOURCES src/GUI/UMANSQtGuiApplication.qrc)
	qt5_add_resources(UI_RESOURCES_RCC ${UI_RESOURCES})
	
	# --- define the project itself
	add_executable(UMANS-GUI src/GUI/main.cpp ${src_GUI} ${UI_RESOURCES_RCC} ${UI_HEADERS})

	# --- the project depends on some libraries
	target_link_libraries(UMANS-GUI Engine Qt5::Core Qt5::Widgets ${OPENGL_LIBRARIES})

	# --- post-build command: copy the required Qt DLLs, to prevent runtime errors
	add_custom_command(TARGET UMANS-GUI POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Qt5::Core> $<TARGET_FILE_DIR:UMANS-GUI>
		COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Qt5::Gui> $<TARGET_FILE_DIR:UMANS-GUI>
		COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Qt5::Widgets> $<TARGET_FILE_DIR:UMANS-GUI>
		)

endif (UMANS_USE_GUI_APPLICATION)

# === Clean up the project's directory structure

foreach(_source IN ITEMS ${src_All})
    get_filename_component(_source_path "${_source}" PATH)
    file(RELATIVE_PATH _source_path_rel "${CMAKE_CURRENT_SOURCE_DIR}/src/" "${_source_path}")
    string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
    source_group("${_group_path}" FILES "${_source}")
endforeach()

