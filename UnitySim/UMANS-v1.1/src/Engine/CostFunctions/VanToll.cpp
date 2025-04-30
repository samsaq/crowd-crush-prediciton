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

#include <CostFunctions/VanToll.h>
#include <core/worldBase.h>
#include <core/agent.h>
#include "core/costFunctionFactory.h"

using namespace std;

float VanToll::GetCost(const Vector2D& velocity, Agent* agent, const WorldBase * world) const
{
	const Vector2D& currentPos = agent->getPosition();
	const Vector2D& currentVel = agent->getVelocity();
	const Vector2D& prefVel = agent->getPreferredVelocity();
	const float prefSpeed = prefVel.magnitude();
	const float speed = velocity.magnitude();

	// compute the time to collision
	const float candidateTTC = ComputeTimeToFirstCollision(currentPos, velocity, agent->getRadius(), agent->getNeighbors(), range_, false);

	// compute the distance to collision, bounded by the viewing distance
	float distanceToCollision = (candidateTTC == MaxFloat ? MaxFloat : candidateTTC * speed);
	if (distanceToCollision > max_distance)
		distanceToCollision = max_distance;

	const float distanceToCollisionFrac = (max_distance - distanceToCollision) / max_distance;

	const float deviationFromPreferredAngle = angle(velocity, prefVel);
	const float deviationFromPreferredSpeed = abs(speed - prefSpeed) / prefSpeed;
	const float deviationFromCurrentAngle = angle(velocity, currentVel);

	// The cost of this velocity is a weighted sum of various factors
	const float Weight_DistanceToCollision = 1;
	const float Weight_DeviationFromPreferredAngle = 1;
	const float Weight_DeviationFromPreferredSpeed = 1;
	const float Weight_DeviationFromCurrentAngle = 1;

	const float candidateCost =
		Weight_DistanceToCollision * (max_distance - distanceToCollision) +
		Weight_DeviationFromPreferredAngle * deviationFromPreferredAngle +
		Weight_DeviationFromPreferredSpeed * deviationFromPreferredSpeed +
		Weight_DeviationFromCurrentAngle * deviationFromCurrentAngle;

	return candidateCost;
}

void VanToll::parseParameters(const CostFunctionParameters & params)
{
	CostFunction::parseParameters(params);
	params.ReadFloat("max_distance", max_distance);
}