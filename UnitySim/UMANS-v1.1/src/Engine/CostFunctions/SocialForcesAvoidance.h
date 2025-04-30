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

#ifndef LIB_SOCIAL_FORCES_H
#define LIB_SOCIAL_FORCES_H

#include <CostFunctions/ObjectInteractionForces.h>

/// @ingroup costfunctions
/// <summary>An implementation of the collision-avoidance force proposed by Helbing and Molnar,
/// in the paper "Social force model for pedestrian dynamics" (1995).</summary>
/// <remarks>
/// This paper defines a force per neighboring agent, and thus the implementation inherits from ObjectInteractionForces.
///
/// ### Notes:
/// <list>
/// <item>This cost function does *not* include a goal-reaching component. 
/// Thus, to bring agents to their goals, you need to combine this cost function with (for example) GoalReachingForce.</item>
/// <item>The <tt>%SocialForces.xml</tt> policy in our example files reproduces the original algorithm, 
/// using two cost functions: SocialForces and GoalReachingForce.</item>
///
/// ### Name in XML files:
/// <tt>%SocialForcesAvoidance</tt>
/// 
/// ### Parameters:
/// <list>
/// <item>dt (float): A time window parameter used in force calculations. It does not have a very intuitive meaning.</item>
/// <item>V0 (float): A scaling factor for all agent-avoidance forces.</item>
/// <item>sigma (float): Determines the steepness of the exponential function defining an agent-avoidance force.
/// Could be interpreted as the distance (in meters) that an agent wants to keep from another agent.</item>
/// <item>U0 (float): A scaling factor for all obstacle-avoidance forces.</item>
/// <item>R (float): Determines the steepness of the exponential function defining an obstacle-avoidance force. 
/// Could be interpreted as the distance (in meters) that an agent wants to keep from obstacles.</item>
/// </list>
/// </remarks>
class SocialForcesAvoidance : public ObjectInteractionForces
{
private:
	float dt = 1;
	float V0 = 2.1f;
	float sigma = 0.3f;
	float U0 = 10;
	float R = 0.2f;

public:
	SocialForcesAvoidance() : ObjectInteractionForces() {}
	virtual ~SocialForcesAvoidance() {}
	const static std::string GetName() { return "SocialForcesAvoidance"; }
	void parseParameters(const CostFunctionParameters & params) override;

protected:
	virtual Vector2D ComputeAgentInteractionForce(const Agent* agent, const PhantomAgent& other) const override;
	virtual Vector2D ComputeObstacleInteractionForce(const Agent* agent, const LineSegment2D& obstacle) const override;
};

#endif //LIB_SOCIAL_FORCES_H
