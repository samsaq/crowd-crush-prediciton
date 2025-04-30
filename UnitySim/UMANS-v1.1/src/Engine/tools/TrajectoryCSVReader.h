/* UMANS: Unified Microscopic Agent Navigation Simulator
** MIT License
** Copyright (C) 2018-2020  Inria Rennes Bretagne Atlantique - Rainbow - Julien Pettré
**
** Permission is hereby granted, free of charge, to any person obtaining
** a copy of this software and associated documentation files (the
** "Software"), to deal in the Software without restriction, including
** without limitation the rights to use, copy, modify, merge, publish,
** distribute, sublicense, and/or sell copies of the Software, and to
** permit persons to whom the Software is furnished to do so, subject
** to the following conditions:
**
** The above copyright notice and this permission notice shall be
** included in all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
** EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
** OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
** NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
** LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
** ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
** CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**
** Contact: crowd_group@inria.fr
** Website: https://project.inria.fr/crowdscience/
** See the file AUTHORS.md for a list of all contributors.
*/

#ifndef LIB_TRAJECTORYCSVREADER_H
#define LIB_TRAJECTORYCSVREADER_H

#include <tools/Trajectory.h>

#include <string>
#include <vector>
#include <mutex>
#include <map>

/// <summary>
/// A class with static functions for loading trajectories from CSV files.
/// </summary>
class TrajectoryCSVReader
{
public:
	/// <summary>Loads a set of agent trajectories from a folder with CSV files.</summary>
	/// <remarks>This function loads all CSV files that have a name in the format "prefix_ID.csv". 
	/// Here, "prefix" can be any prefix, and "ID" must be a non-negative integer that is unique within that folder. 
	/// Files that do not meet these constraints will not be loaded.</remarks>
	/// <returns>A set of agent IDs and trajectories. The result is empty if the loading fails.</returns>
	static AgentTrajectories ReadTrajectoriesFromCSVFolder(const std::string& folderName);

	/// <summary>Loads a trajectory from a single CSV file.</summary>
	static Trajectory ReadTrajectoryFromCSVFile(const std::string& filename);
};

#endif // LIB_TRAJECTORYCSVREADER_H
