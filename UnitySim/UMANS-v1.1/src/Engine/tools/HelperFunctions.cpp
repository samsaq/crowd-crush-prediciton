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

#include <tools/HelperFunctions.h>

#include <sstream>
#include <iomanip>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <math.h>       /* fmod */
#endif

namespace HelperFunctions
{
	Timestamp GetCurrentTime()
	{
		return std::chrono::high_resolution_clock::now();
	}

	double GetIntervalMilliseconds(const Timestamp& startTime, const Timestamp& endTime)
	{
		return std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count() / 1000000.0;
	}
	
	std::vector<std::string> SplitString(const std::string& str, const char delim)
	{
		std::vector<std::string> result;

		size_t offset = 0;
		while (offset != std::string::npos)
		{
			size_t index = str.find(delim, offset);
			if (index == std::string::npos)
			{
				result.push_back(str.substr(offset));
				offset = index;
			}
			else
			{
				const auto& part = str.substr(offset, index - offset);
				if (!part.empty())
					result.push_back(part);
				offset = index + 1;
			}
		}

		return result;
	}

	double ParseDouble(const std::string& str)
	{
		std::istringstream iss(str);

		double val;
		if (!(iss >> val))
			throw std::string("Could not parse string to double.");
		return val;
	}

	float ParseFloat(const std::string& str)
	{
		std::istringstream iss(str);

		float val;
		if (!(iss >> val))
			throw std::string("Could not parse string to float.");
		return val;
	}

	int ParseInt(const std::string& str)
	{
		std::istringstream iss(str);

		int val;
		if (!(iss >> val))
			throw std::string("Could not parse string to int.");
		return val;
	}

	bool ParseBool(const std::string& str)
	{
		if (str == "true")
			return true;
		if (str == "false")
			return false;

		throw std::string("Could not parse string to bool.");
	}

	std::string ToString(int value)
	{
		std::stringstream ss;
		ss << value;
		return ss.str();
	}

	std::string ToStringWithLeadingZeros(int value, int length)
	{
		std::stringstream ss;
		ss << std::setw(length) << std::setfill('0') << value;
		return ss.str();
	}

	bool DirectoryExists(const std::string &dirname)
	{
		struct stat info;

		if (stat(dirname.c_str(), &info) != 0)
			return false;
		else if (info.st_mode & S_IFDIR)
			return true;
		else
			return false;
	}

	bool CreateDirectoryIfNonExistent(const std::string& dirname)
	{
		// if dirname contains a sequence of folders, try to create it step by step
		size_t index = dirname.find_first_of("/");
		do
		{
			const std::string& dirpart = dirname.substr(0, index);

			// if the directory does not yet exist, try to create it
			if (!DirectoryExists(dirpart))
			{
				int nError = 0;
#if defined(_WIN32)
				nError = _mkdir(dirpart.c_str()); // can be used on Windows
#else 
				mode_t nMode = 0733;
				nError = mkdir(dirpart.c_str(), nMode); // can be used on non-Windows
#endif
				if (nError != 0)
					return false;
			}

			index = dirname.find_first_of("/", index+1);
		} while (index != std::string::npos);

		return true;
	}
};