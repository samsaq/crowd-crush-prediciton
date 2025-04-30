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

#ifndef LIB_POWER_LAW_H
#define LIB_POWER_LAW_H

#include <CostFunctions/ObjectInteractionForces.h>
#include <string>

/// @ingroup costfunctions
/// <summary>An implementation of the collision-avoidance force proposed by %Karamouzas et al.,
/// in the paper "Universal Power Law Governing Pedestrian Interactions" (2014).</summary>
/// <remarks>
/// This paper defines a force per neighboring agent, and thus the implementation inherits from ObjectInteractionForces.
///
/// ### Notes: 
/// <list>
/// <item>This cost function does *not* include a goal-reaching component. 
/// Thus, to bring agents to their goals, you need to combine this cost function with (for example) GoalReachingForce.</item>
/// <item>The <tt>%PowerLaw.xml</tt> policy in our example files reproduces the original algorithm.
/// using two cost functions: PowerLaw and GoalReachingForce.</item>
/// <item><strong>WARNING:</strong> This cost function does not yet support static obstacles, so any obstacles will be ignored for now.
/// This feature should be added in the future. Unfortunately, the original paper does not describe how obstacles are handled, 
/// and its original implementation cannot be freely used.</item>
/// </list>
/// 
/// ### Name in XML files:
/// <tt>%PowerLaw</tt>
///
/// ### Parameters:
/// <list>
/// <item>k (float): A scaling factor for the overall force.</item>
/// <item>tau0 (float): A time threshold, which could be interpreted as the "maximum time-to-collision of interest".</item>
/// </list>
/// </remarks>
class PowerLaw : public ObjectInteractionForces
{
private:
	float k = 1.5f;
	float tau0 = 3;

public:
	PowerLaw() : ObjectInteractionForces() {}
	virtual ~PowerLaw() {}
	const static std::string GetName() { return "PowerLaw"; }

	void parseParameters(const CostFunctionParameters & params) override;

protected:
	virtual Vector2D ComputeAgentInteractionForce(const Agent* agent, const PhantomAgent& other) const override;
	virtual Vector2D ComputeObstacleInteractionForce(const Agent* agent, const LineSegment2D& obstacle) const override;
};

#endif //LIB_POWER_LAW_H
