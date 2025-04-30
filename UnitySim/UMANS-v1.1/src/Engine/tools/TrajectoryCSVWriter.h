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

#ifndef LIB_TRAJECTORYCSVWRITER_H
#define LIB_TRAJECTORYCSVWRITER_H

#include <tools/Trajectory.h>

#include <string>
#include <vector>
#include <mutex>
#include <map>

typedef std::map<size_t, TrajectoryPoint> AgentTrajectoryPoints;
typedef std::map<size_t, Trajectory> AgentTrajectories;

/// <summary>A class for writing agent trajectories to CSV files in a specified folder.</summary>
class TrajectoryCSVWriter
{
private:
    std::mutex mtx_;
    std::string dirname_;
	AgentTrajectories pos_log_;

	/// <summary>Whether or not this TrajectoryCSVWriter immediately calls Flush() at the end of each call to AppendAgentPositions().</summary>
	bool flushImmediately;

public:
	/// <summary>Creates a TrajectoryCSVWriter object.</summary>
	/// <param name="flushImmediately">Whether or not this TrajectoryCSVWriter should immediately call Flush() at the end of each call to AppendAgentPositions().</param>
    TrajectoryCSVWriter(bool flushImmediately) : flushImmediately(flushImmediately) {}

	/// <summary>Tries to set the directory to which CSV output files will be written.
	/// If this directory does not yet exist, the program will try to create it, but this operation might fail.</summary>
	/// <param name="dirname">The path to the desired output directory.</param>
	/// <returns>true if the directory was successfully set (and created if necessary); 
	/// false otherwise, i.e. if the folder did not exist and could not be created for some reason.</returns>
    bool SetOutputDirectory(const std::string &dirname);

	/// <summary>Appends a set of agent positions to the output buffer. 
	/// Note: if the flushImmediately parameter is false, the result is not yet written to a file, and you need to call Flush() yourself.</summary>
	/// <param name="poss">A list of agent positions, ordered by agent ID in a map.</param>
	/// <param name="t">The current simulation time.</param>
    void AppendAgentData(const AgentTrajectoryPoints& data);

	/// <summary>Writes all buffered output to CSV files, and then cleans the buffer.</summary>
	/// <remarks>Call this method whenever you have finished gathering trajectory data via AppendAgentPositions().</remarks>
	/// <returns>true if the output was successfully written; 
	/// false otherwise, e.g. if the output folder was never specified via SetOutputDirectory().</returns>
	bool Flush();

	/// <summary>Cleans up this TrajectoryCSVWriter for removal. This includes a final call to flush().</summary>
	~TrajectoryCSVWriter();
};

#endif // LIB_TRAJECTORYCSVWRITER_H
