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

#include <CostFunctions/ObjectInteractionForces.h>
#include <core/agent.h>
#include <core/worldBase.h>

Vector2D ObjectInteractionForces::ComputeForce(Agent* agent, const WorldBase * world) const
{
	const Vector2D& Position = agent->getPosition();
	const float rangeSquared = range_ * range_;

	// loop over all neighbors; sum up the forces for all neighbors that are in range

	// - agents
	Vector2D AgentForces(0, 0);
	const auto& neighbors = agent->getNeighbors();
	for (const PhantomAgent& other : neighbors.first)
	{
		if (other.GetDistanceSquared() <= rangeSquared)
			AgentForces += ComputeAgentInteractionForce(agent, other);
	}

	// - obstacles
	Vector2D ObstacleForces(0, 0);
	for (const LineSegment2D& obs : neighbors.second)
	{
		// We know that obstacles are always closed polygons, whose boundary points are given in counter-clockwise order.

		// 1. Ignore an obstacle segment if an agent lies on the wrong side of it. In this case, either the agent is inside the obstacle, or another segment is more relevant.
		if (isPointLeftOfLine(Position, obs.first, obs.second))
			continue;

		// 2. Ignore an obstacle segment if the nearest point is its second endpoint. This prevents double forces at obstacle corners.
		const Vector2D& nearest = nearestPointOnLine(Position, obs.first, obs.second, true);
		if (nearest == obs.second)
			continue;

		if (distanceSquared(Position, nearest) <= rangeSquared)
			ObstacleForces += ComputeObstacleInteractionForce(agent, obs);
	}

	return AgentForces + ObstacleForces;
}
