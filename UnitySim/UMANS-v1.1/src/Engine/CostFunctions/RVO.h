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

#ifndef LIB_RVO_H
#define LIB_RVO_H

#include <core/costFunction.h>

/// @ingroup costfunctions
/// <summary>An implementation of the full navigation function proposed by van den Berg et al., 
/// in the paper "Reciprocal velocity obstacles for real-time multi-agent navigation" (2008).</summary>
/// <remarks>
/// ### Notes: 
/// <list>
/// <item>This cost function already includes a goal-reaching component.</item>
/// <item>Although the %RVO paper describes the navigation problem in a closed mathematical form, 
/// the authors eventually propose a sampling-based implementation. 
/// Thus, our implementation does not contain a closed-form solution for finding the optimal velocity.</item>
/// <item>This is **not** the %ORCA method (sometimes referred to as "RVO2") that uses linear programming.
/// If you want to use that, use the ORCA cost function instead.</item>
/// <item>The <tt>%RVO.xml</tt> policy in our example files reproduces the original algorithm.</item>
/// </list>
/// 
/// ### Name in XML files:
/// <tt>%RVO</tt>
///
/// ### Parameters: 
/// <list>
/// <item>w: A weight for one of the two main components of the cost function.</item>
/// </list>
/// </remarks>
class RVO : public CostFunction
{
private:
	float w = 1; // weight in the cost function

public:
	RVO() : CostFunction() {}
	virtual ~RVO() {}
	const static std::string GetName() { return "RVO"; }

	virtual float GetCost(const Vector2D& velocity, Agent* agent, const WorldBase * world) const override;

	void parseParameters(const CostFunctionParameters & params) override;
};

#endif //LIB_RVO_H
