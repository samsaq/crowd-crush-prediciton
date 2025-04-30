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

#include <tools/TrajectoryCSVReader.h>
#include <tools/HelperFunctions.h>

#include <fstream>
#include <filesystem>

AgentTrajectories TrajectoryCSVReader::ReadTrajectoriesFromCSVFolder(const std::string& foldername)
{
	AgentTrajectories result;

	// verify that this is a directory
	if (!HelperFunctions::DirectoryExists(foldername))
		return result;

	// loop over the folder (non-recursively)
	using namespace std::filesystem;
	directory_iterator end_itr;
	for (directory_iterator itr(foldername); itr != end_itr; ++itr)
	{
		// check if this is a file
		if (is_directory(itr->status()))
			continue;

		// check if it has the CSV extension
		const std::string& filename = itr->path().string();
		if (filename.substr(filename.length()-4) != ".csv")
			continue;

		// check if an agent ID can be obtained from the filename
		const size_t index_underscore = filename.find_last_of("_");
		if (index_underscore == std::string::npos)
			continue;
		const std::string& number = filename.substr(index_underscore + 1, filename.length() - index_underscore - 5);
		try
		{
			size_t agentID = HelperFunctions::ParseInt(number);
			if (agentID >= 0)
			{
				// read a single trajectory; add it to the result
				result.insert({ agentID, ReadTrajectoryFromCSVFile(filename) });
			}
		}
		catch (std::exception&) {}

	}

	return result;
}

Trajectory TrajectoryCSVReader::ReadTrajectoryFromCSVFile(const std::string& filename)
{
	// open the file
	std::ifstream fs;
	fs.open(filename);
	if (!fs.is_open())
		throw std::string("CSV file not found.");

	Trajectory result;
	std::string line;
	while (std::getline(fs, line))
	{
		const auto& parts = HelperFunctions::SplitString(line, ',');
		if (parts.size() < 3)
			continue;

		TrajectoryPoint point;

		// read the time
		point.time = HelperFunctions::ParseDouble(parts[0]);

		// read the position
		point.position.x = HelperFunctions::ParseFloat(parts[1]);
		point.position.y = HelperFunctions::ParseFloat(parts[2]);

		if (parts.size() >= 5)
		{
			// read the orientation
			point.orientation.x = HelperFunctions::ParseFloat(parts[3]);
			point.orientation.y = HelperFunctions::ParseFloat(parts[4]);
		}
		else
		{
			// leave the orientation empty
			point.orientation.x = 0;
			point.orientation.y = 0;
		}

		// add this point to the result
		result.push_back(point);
	}

	return result;
}