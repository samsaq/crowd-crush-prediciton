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

#ifndef LIB_PLEDESTRIANS_H
#define LIB_PLEDESTRIANS_H

#include <core/costFunction.h>

/// @ingroup costfunctions
/// <summary>An implementation of the full navigation fuction proposed by Guy et al.,
/// in the paper "PLEdestrians: A least-effort approach to crowd simulation" (2010).</summary>
/// <remarks>
/// ### Notes: 
/// <list>
/// <item>This cost function already includes a goal-reaching component.</item>
/// <item>The original paper presents a closed-form solution for finding the optimal velocity.
/// Our implementation (currently) does not contain this closed-form solution.
/// To obtain an approximation of the original method, use this cost function in a sampling-based policy.</item>
/// <item>The <tt>%PLEdestrians.xml</tt> policy in our example files gives a good approximation of the original algorithm.</item>
/// </list>
/// 
/// ### Name in XML files:
/// <tt>%PLEdestrians</tt>
///
/// ### Parameters: 
/// <list>
/// <item>w_a, w_b (float): two weights for the main components of the cost function.</item>
/// <item>t_min, t_max (float): two time thresholds.</item>
/// </list>
/// </remarks>
class PLEdestrians : public CostFunction
{
private:
	float w_a = 2.23f; // weight in the cost function
	float w_b = 1.26f;
	float t_min = 0.5f;
	float t_max = 3;

public:
	PLEdestrians() : CostFunction() {}
	virtual ~PLEdestrians() {}
	const static std::string GetName() { return "PLEdestrians"; }

	virtual float GetCost(const Vector2D& velocity, Agent* agent, const WorldBase * world) const override;

	void parseParameters(const CostFunctionParameters & params) override;
};

#endif //LIB_PLEDESTRIANS_H
