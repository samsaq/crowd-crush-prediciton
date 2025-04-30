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

#include <core/costFunctionFactory.h>

#include <CostFunctions/FOEAvoidance.h>
#include <CostFunctions/GenericCost.h>
#include <CostFunctions/GoalReachingForce.h>
#include <CostFunctions/Karamouzas.h>
#include <CostFunctions/Moussaid.h>
#include <CostFunctions/ORCA.h>
#include <CostFunctions/Paris.h>
#include <CostFunctions/PLEdestrians.h>
#include <CostFunctions/PowerLaw.h>
#include <CostFunctions/RandomFunction.h>
#include <CostFunctions/RVO.h>
#include <CostFunctions/SocialForcesAvoidance.h>
#include <CostFunctions/TtcaDca.h>
#include <CostFunctions/VanToll.h>

CostFunctionFactory::Registry CostFunctionFactory::registry = CostFunctionFactory::Registry();

void CostFunctionFactory::RegisterAllCostFunctions()
{
	// do the registration only once
	if (!registry.empty())
		return;

	registerCostFunction<FOEAvoidance>();
	registerCostFunction<GenericCost>();
	registerCostFunction<GoalReachingForce>();
	registerCostFunction<Karamouzas>();
	registerCostFunction<Moussaid>();
	registerCostFunction<ORCA>();
	registerCostFunction<PowerLaw>();
	registerCostFunction<RandomFunction>();
	registerCostFunction<SocialForcesAvoidance>();
	registerCostFunction<TtcaDca>();
	registerCostFunction<RVO>();
	registerCostFunction<Paris>();
	registerCostFunction<PLEdestrians>();
	registerCostFunction<VanToll>();
}

void CostFunctionFactory::ClearRegistry()
{
	registry.clear();
}