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

#include <CostFunctions/ORCA.h>
#include <core/worldBase.h>
#include "../3rd-party/ORCA/ORCASolver.h"

const ORCALibrary::Solution& ORCA::GetOrcaSolutionForAgent(Agent* agent, const WorldBase* world) const
{
	// check if the agent needs to run ORCA again
	const auto& cachedSolution = agent->GetOrcaSolution();
	if (cachedSolution.currentSimulationTime < world->GetCurrentTime())
	{
		// run ORCA and store the solution in the agent
		ORCALibrary::Solver solver;
		solver.solveOrcaProgram(*agent, timeHorizon, (float)world->GetCurrentTime(), world->GetDeltaTime(), agent->getNeighbors(), range_, agent->GetOrcaSolution());
	}

	// return the result
	return agent->GetOrcaSolution();
}

inline float getSignedDistanceToOrcaLine(const Vector2D& velocity, const ORCALibrary::Line& line)
{
	//return (line.point - velocity).dot(Vector2D(line.direction.y, -line.direction.x));

	const Vector2D& pMinV = line.point - velocity;
	return line.direction.x * pMinV.y - line.direction.y * pMinV.x;
}

float ORCA::GetCost(const Vector2D& velocity, Agent* agent, const WorldBase * world) const
{
	const Vector2D& currentVelocity = agent->getVelocity();

	// compute or re-use the ORCA solution for this agent
	const auto& orcaSolution = GetOrcaSolutionForAgent(agent, world);

	// Find the maximum distance by which this velocity exceeds any ORCA plane.
	// The "getSignedDistanceToOrcaLine" function returns a positive number if the ORCA constraint is violated.
	float maxDistance = -MaxFloat;

	for (const auto& orcaLine : orcaSolution.orcaLines)
		maxDistance = std::max(maxDistance, getSignedDistanceToOrcaLine(velocity, orcaLine));

	// There are three possible cases:
	//
	// a) ORCA has a solution, and this velocity is inside the solution space.
	//    In this case, maxDistance has to be <= 0, because the velocity is on the correct side of all ORCA lines.
	//    For such "allowed" velocities, ORCA uses the difference to vPref as the cost.
	//
	if (maxDistance <= 0)
	{
		// the cost is the difference to vPref
		return (velocity - agent->getPreferredVelocity()).magnitude();
	}
	//
	// b) ORCA has a solution, but this velocity is not inside the solution space.
	//    In this case, according to "the real ORCA method", we should return an infinite cost to prevent this velocity from being chosen.
	//    However, in the context of sampling and gradients, it is better to return a finite value, to distinguish between "bad" and "even worse" velocities.
	//    Returning maxDistance is a good option, but we should add a sufficiently large constant, so that velocities inside the ORCA solution space will be preferred.
	//
	// c) ORCA does not have a solution, and we need to use ORCA's "backup function", which is just maxDistance.
	//    In this case, it does not hurt to add the same large constant as in case (b).
	//
	// In short, both cases can actually use the same cost function:

	return maxDistance + 2 * agent->getMaximumSpeed();
}

Vector2D ORCA::GetGlobalMinimum(Agent* agent, const WorldBase* world) const
{
	// compute or re-use the ORCA solution for this agent
	const auto& orcaSolution = GetOrcaSolutionForAgent(agent, world);

	// return the optimal velocity that was computed by ORCA
	return orcaSolution.velocity;
}

void ORCA::parseParameters(const CostFunctionParameters & params)
{
	CostFunction::parseParameters(params);
	params.ReadFloat("timeHorizon", timeHorizon);
}