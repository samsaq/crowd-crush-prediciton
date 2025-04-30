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

#include <tools/TrajectoryCSVWriter.h>
#include <tools/HelperFunctions.h>

#include <fstream>

bool TrajectoryCSVWriter::SetOutputDirectory(const std::string &dirname)
{
	// if necessary, append a slash to the directory path
	if (dirname.back() == '/')
		this->dirname_ = dirname;
	else
		this->dirname_ = dirname + '/';

	// try to create the directory, and return if the directory exists
	HelperFunctions::CreateDirectoryIfNonExistent(this->dirname_);
	return HelperFunctions::DirectoryExists(this->dirname_);
}

bool TrajectoryCSVWriter::Flush()
{
	if (dirname_.empty())
		return false;

    mtx_.lock();
	AgentTrajectories pos_log_copy = pos_log_;
    pos_log_.clear();
    mtx_.unlock();

    for (auto& data : pos_log_copy)
	{
        std::fstream file_i;
        size_t id = data.first;
        std::string file_name = dirname_ + "output_" + std::to_string(id) + ".csv";
        file_i.open(file_name, std::ios::app);
		for (const TrajectoryPoint& record : data.second)
		{
			file_i << record.time << ","
				<< record.position.x << "," << record.position.y << ","
				<< record.orientation.x << "," << record.orientation.y << "\n";
		}
        file_i.close();
        data.second.clear();
    }

	return true;
}
void TrajectoryCSVWriter::AppendAgentData(const AgentTrajectoryPoints& data)
{
    mtx_.lock();
    for (const auto& d : data)
        pos_log_[d.first].push_back(d.second);
    mtx_.unlock();

	if (flushImmediately)
		Flush();
}

TrajectoryCSVWriter::~TrajectoryCSVWriter()
{
	Flush();
}