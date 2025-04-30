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

#ifndef _COST_FUNCTION_FACTORY_H
#define _COST_FUNCTION_FACTORY_H

#include <core/costFunction.h>
#include <functional>
#include <map>
#include <string>
#include <iostream>

/// <summary>A static class that can create new instances of cost functions upon request.</summary>
class CostFunctionFactory
{
public:
	typedef std::map<std::string, std::function<CostFunction*()>> Registry;

private:
	/// <summary>A mapping from names to CostFunction creators.</summary>
	static Registry registry;

private:
	/// <summary>Adds the given CostFunction class type to the registry, and stores it under the name specified in that class's GetName() function. 
	/// After completion of this method, CreateCostFunction() will be able to create instances of this class type.</summary>
	template<typename CostFunctionType> static void registerCostFunction()
	{
		const std::string& name = CostFunctionType::GetName();
		if (registry.count(name) > 0)
		{
			std::cerr << "Error: cost function " << name << " has already been registered." << std::endl;
			return;
		}
		registry[name] = [] { return new CostFunctionType(); };
	}

public:
	/// <summary>Registers all possible types of cost functions that can exist. 
	/// Call this method once before you start using CreateCostFunction().</summary>
	static void RegisterAllCostFunctions();

	/// <summary>Clears the list of registered cost functions. 
	/// Call this method when you destroy the Simulator.</summary>
	static void ClearRegistry();

	/// <summary>Creates and returns a new CostFunction instance whose name matches the given value.</summary>
	/// <remarks>This method looks through the registry created in RegisterAllCostFunctions().</remarks>
	/// <param name="name">The name of the cost function to create.</param>
	/// <returns>A pointer to a new instance of the CostFunction subclass whose GetName() matches the "name" parameter.</returns>
	static CostFunction* CreateCostFunction(const std::string& name)
	{
		if (registry.count(name) == 0)
		{
			std::cerr << "Error: cost function " << name << " does not exist. The following cost functions are known: ";
			for (auto &elm : registry)
				std::cerr << ", " << elm.first;
			std::cerr << "." << std::endl;
			return nullptr;
		}

		return registry[name]();
	}
};

#endif
