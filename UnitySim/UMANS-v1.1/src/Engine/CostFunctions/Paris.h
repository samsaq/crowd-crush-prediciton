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

#ifndef LIB_PARIS_H
#define LIB_PARIS_H

#include <core/costFunction.h>

/// @ingroup costfunctions
/// <summary>(Work in progress) A re-interpretation of the collision-avoidance model proposed by %Paris et al., 
/// in the paper "Pedestrian reactive navigation for crowd simulation - A predictive approach" (2007).</summary>
/// <remarks>
/// ### Notes: 
/// <list>
/// <item>This cost function already includes a goal-reaching component.</item>
/// <item>The original description of this navigation model is not entirely clear, and its source code cannot be retrieved. 
/// Therefore, this class is merely *our* interpretation of it, translated to the domain of cost functions.
/// There is no guarantee that this interpretation perfectly matches the authors' original intentions.</item>
/// <item>The <tt>%Paris.xml</tt> policy in our example files attempts to reproduce the original algorithm, but (again) without any guarantees.</item>
/// <item><strong>WARNING:</strong> This cost function does not yet support static obstacles, so any obstacles will be ignored for now.
/// This feature should be added in the future. Unfortunately, the original paper does not clearly define what to do with obstacles.</item>
/// </list>
///
/// ### Name in XML files:
/// <tt>%Paris</tt>
///
/// ### Parameters: 
/// <list>
/// <item>w_a (float): A weight for the "collision avoidance" component of the cost function. 
/// Determines the ratio between the "collision avoidance" and "preferred motion" components.</item>
/// <item>w_b (float): A scalar that determines how the "collision avoidance" importance of a neighbor decreases as time-to-collision increases.
/// <item>t_max (float): The maximum time-to-collision to consider.</item>
/// </list>
/// </remarks>
class Paris : public CostFunction
{
private:
	struct NeighborAvoidanceRange
	{
		float t1, t2, s1, s2;
		NeighborAvoidanceRange(float t1, float t2, float s1, float s2) : t1(t1), t2(t2), s1(s1), s2(s2) {}
	};

	typedef std::vector<NeighborAvoidanceRange> NeighborAvoidanceRangeList;

	float w_a = 0.5f; // weight in the cost function
	float w_b = 0;
	float t_max = 8; // maximum time to collision

public:
	Paris() : CostFunction() {}
	virtual ~Paris() {}
	const static std::string GetName() { return "Paris"; }

	virtual float GetCost(const Vector2D& velocity, Agent* agent, const WorldBase * world) const override;

	void parseParameters(const CostFunctionParameters & params) override;

private:
	std::pair<float, float> ComputeT1andT2(const Vector2D& origin, const Vector2D& velocity, const Vector2D& point, const float radius) const;
	NeighborAvoidanceRange ComputeNeighborAvoidanceRange(const Vector2D& direction, const Agent* agent, const PhantomAgent& neighbor) const;
	NeighborAvoidanceRangeList ComputeAllNeighborAvoidanceRanges(const Vector2D& direction, const Agent* agent) const;

	float GetDirectionCost(const Vector2D& direction, const Agent* agent, const NeighborAvoidanceRangeList& avoidanceTimes) const;
	float GetSpeedDeviationCost(const float speed, const Vector2D& direction, const Agent* agent, const NeighborAvoidanceRangeList& avoidanceRanges) const;

	float CSpeed(const float preferredSpeed, const NeighborAvoidanceRange& avoidance) const;
	float CPred(const NeighborAvoidanceRange& avoidance) const;
	float CAngle(const Vector2D& direction, const Vector2D& preferredVelocity) const;
};

#endif //LIB_PARIS_H
