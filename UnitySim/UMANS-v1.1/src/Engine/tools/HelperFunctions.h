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

#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <vector>
#include <string>
#include <chrono>

namespace HelperFunctions
{
	typedef std::chrono::time_point<std::chrono::high_resolution_clock> Timestamp;

	/// <summary>Returns the current system time.</summary>
	Timestamp GetCurrentTime();
	/// <summary>Returns the number of milliseconds between two time stamps.</summary>
	double GetIntervalMilliseconds(const Timestamp& startTime, const Timestamp& endTime);
	
	/// <summary>Splits a string into parts according to a given delimiter. 
	/// The delimiter itself will be excluded from these parts.</summary>
	/// <param name="str">The string to split.</param>
	/// <param name="delim">The delimiter to use for splitting.</param>
	/// <returns>A list of substrings, each containing a part of the input string between two subsequent delimiters.
	/// The delimiter itself is not included in any of these parts.</returns>
	std::vector<std::string> SplitString(const std::string& str, const char delim);

	/// <summary>Tries to convert a string to a double.</summary>
	double ParseDouble(const std::string& str);
	/// <summary>Tries to convert a string to a float.</summary>
	float ParseFloat(const std::string& str);
	/// <summary>Tries to convert a string to an int.</summary>
	int ParseInt(const std::string& str);
	/// <summary>Tries to convert a string to a bool.</summary>
	bool ParseBool(const std::string& str);

	/// <summary>Converts an integer to a string.</summary>
	std::string ToString(int val);
	/// <summary>Converts an integer to a string with a given minimum length.
	/// If the integer is shorter than that, the string will be augmented with leading 0s.</summary>
	std::string ToStringWithLeadingZeros(int value, int length);

	template <typename T> inline T Clamp(T val, T minValue, T maxValue)
	{
		if (val < minValue)
			return minValue;

		if (val > maxValue)
			return maxValue;

		return val;
	}

	/// <summary>Checks and returns whether a given directory exists.</summary>
	/// <param name="dirname">The name of a directory, either absolute or relative.</param>
	/// <returns>true if the directory exists; false otherwise.</returns>
	bool DirectoryExists(const std::string& dirname);

	/// <summary>Tries to create a directory if it does not yet exist.</summary>
	/// <param name="dirname">The name of a directory, either absolute or relative.</param>
	/// <returns>true if the directory either was successfully created or already existed; false otherwise.</returns>
	bool CreateDirectoryIfNonExistent(const std::string& dirname);
};

#endif //HELPER_FUNCTIONS_H