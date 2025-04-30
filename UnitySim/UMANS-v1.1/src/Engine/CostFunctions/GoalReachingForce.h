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

#ifndef LIB_GOAL_REACHING_H
#define LIB_GOAL_REACHING_H

#include <CostFunctions/ForceBasedFunction.h>

/// @ingroup costfunctions
/// <summary>A force that attracts the agents towards its goal, as used by the Social Forces method (1995) and many other force-based methods.</summary>
/// <remarks>
/// ### Definition:
/// F = (agent.preferredVelocity - agent.currentVelocity) / max(dt, policy.relaxationTime).
/// We disallow relaxation times smaller than dt (the simulation time step) because this would yield undesired behavior.
/// ### Name in XML files:
/// <tt>%GoalReachingForce</tt></remarks>
class GoalReachingForce : public ForceBasedFunction
{
public:
	GoalReachingForce() : ForceBasedFunction() { range_ = 0; }
	virtual ~GoalReachingForce() {}
	const static std::string GetName() { return "GoalReachingForce"; }

protected:
	/// <summary>Computes a 2D force vector that steers the agent towards its preferred velocity.</summary>
	/// <param name="agent">The agent for which a force is requested.</param>
	/// <param name="world">The world in which the simulation takes place.</param>
	/// <returns>A force vector that steers the agent towards its preferred velocity.</returns>
	virtual Vector2D ComputeForce(Agent* agent, const WorldBase* world) const override;
};

#endif //LIB_GOAL_REACHING_H
