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

#include <CostFunctions/Moussaid.h>
#include <core/worldBase.h>
#include <core/agent.h>

using namespace std;

float Moussaid::getDistanceToCollisionAtPreferredSpeed(const Vector2D& direction, const Agent* agent) const
{
	const float prefSpeed = agent->getPreferredSpeed();
	const Vector2D& velocityWithPrefSpeed = direction * prefSpeed;

	// compute the time to collision at this velocity
	float ttc = ComputeTimeToFirstCollision(agent->getPosition(), velocityWithPrefSpeed, agent->getRadius(), agent->getNeighbors(), range_, false);

	// convert to the distance to collision, clamped to a maximum distance
	float distanceToCollision = (ttc == MaxFloat ? MaxFloat : ttc * prefSpeed);
	if (distanceToCollision > d_max)
		distanceToCollision = d_max;

	return distanceToCollision;
}

float Moussaid::GetCost(const Vector2D& velocity, Agent* agent, const WorldBase * world) const
{
	const float speed = velocity.magnitude(); 
	const float prefSpeed = agent->getPreferredSpeed();

	// compute the distance to collision at maximum speed
	const auto& dir = velocity.getnormalized();
	float DC = getDistanceToCollisionAtPreferredSpeed(dir, agent);

	// compute the cost for this direction, assuming maximum speed
	float K = d_max * d_max + DC * DC - 2 * d_max * DC*cosAngle(dir, agent->getPreferredVelocity());
	K += 1;

	// compute the optimal speed for this direction
	float S = std::min(prefSpeed, DC / agent->getPolicy()->getRelaxationTime());

	// return a cost that penalizes the difference with the optimal speed
	return (float)(K * (1 + pow((S - speed) / S, 2.0)));
}

void Moussaid::parseParameters(const CostFunctionParameters & params)
{
	CostFunction::parseParameters(params);
	params.ReadFloat("d_max", d_max);
}