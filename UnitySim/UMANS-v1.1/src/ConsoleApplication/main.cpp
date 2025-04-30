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

#include <core/crowdSimulator.h>

#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

void printBasicInfo()
{
	std::cout
		<< "-----------------------------------------------------" << std::endl
		<< "UMANS: Unified Microscopic Agent Navigation Simulator" << std::endl
		<< "Copyright(C) 2018 - 2020 Inria Rennes Bretagne Atlantique - Rainbow - Julien Pettre" << std::endl
		<< "-----------------------------------------------------" << std::endl << std::endl;
}

void printUsageInfo(const std::string& programName)
{
	std::cout
		<< "Usage: " << programName << " -i [-o] [-t]" << std::endl
		<< "  -i (or -input)     = An XML file describing the simulation scenario to run." << std::endl
		<< "                       For help on creating scenario files, please see the UMANS documentation." << std::endl
		<< "  -o (or -output)    = (optional) Name of a folder to which the simulation output will be written." << std::endl
		<< "                       The program will write a CSV file for each agent's trajectory." << std::endl
		<< "                       If you omit this, the program will run faster, but no results will be saved." << std::endl
		<< "  -t (or -nrThreads) = (optional, default=1) The number of parallel threads to use." << std::endl << std::endl;
}

int main( int argc, char * argv[] )
{
	printBasicInfo();
	
	std::string configFile = "", outputFolder = "";
	int nrThreads = 1;

	// parse the arguments one by one
	for (int i = 1; i + 1 < argc; i += 2)
	{
		std::string paramName(argv[i]);
		std::string paramValue(argv[i + 1]);

		if (paramName[0] != '-')
		{
			std::cerr << "Input error: " << paramName << " is not a valid parameter name." << std::endl;
			printUsageInfo(argv[0]);
			return -1;
		}

		if (paramName == "-i" || paramName == "-input")
			configFile = paramValue;
		else if (paramName == "-o" || paramName == "-output")
			outputFolder = paramValue;
		else if (paramName == "-t" || paramName == "-nrThreads")
			nrThreads = atoi(paramValue.c_str());
	}

	// input file is mandatory
	if (configFile == "")
	{
		std::cerr << "Input error: Please specify an input scenario." << std::endl;
		printUsageInfo(argv[0]);
		return -1;
	}

	// number of threads must be at least 1
	if (nrThreads < 1)
	{
		std::cerr << "Input error: Please specify a positive number of threads." << std::endl;
		printUsageInfo(argv[0]);
		return -1;
	}

	// output folder may be empty; if so, show a warning
	if (outputFolder == "")
	{
		std::cout << "Input warning: No output folder specified." << std::endl
			<< "The program will not write any simulation results to CSV files." << std::endl;
	}
	
	// run the simulation
	CrowdSimulator* cs = CrowdSimulator::FromConfigFile(configFile);
	if (cs == nullptr) // the simulation could not be loaded (for reasons already written to the console)
		return -1;

	cs->GetWorld()->SetNumberOfThreads(nrThreads);
	if (outputFolder != "")
		cs->StartCSVOutput(outputFolder, false); // false = don't save any files until the simulation ends

	// run the full simulation; show a progress bar and measure the time
	cs->RunSimulationUntilEnd(true, true);

	delete cs;
	return 0;
}
