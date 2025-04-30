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

#include <CostFunctions/RVO.h>
#include <core/worldBase.h>
#include <core/agent.h>

using namespace std;

float RVO::GetCost(const Vector2D& velocity, Agent* agent, const WorldBase * world) const
{ 
	// disallow high speeds
	if (velocity.magnitude() > agent->getMaximumSpeed())
		return MaxFloat;
	
	const float radius = agent->getRadius();
	const Vector2D& position = agent->getPosition();
	const Vector2D& RVOVelocity = 2 * velocity - agent->getVelocity();
	
	const Vector2D& vDiff = agent->getPreferredVelocity() - velocity;

	// compute the smallest time to collision among all neighboring agents
	float minTTC = ComputeTimeToFirstCollision(position, RVOVelocity, radius, agent->getNeighbors(), range_, true);

	return w / minTTC + vDiff.magnitude();
}

void RVO::parseParameters(const CostFunctionParameters & params)
{
	CostFunction::parseParameters(params);
	params.ReadFloat("w", w);
}