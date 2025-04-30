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

#ifndef LIB_FOE_AVOIDANCE_H
#define LIB_FOE_AVOIDANCE_H

#include <core/costFunction.h>

/// @ingroup costfunctions
/// <summary>An implementation of the collision-avoidance cost function proposed by Lopez et al.,
/// in the paper "Character navigation in dynamic environments based on optical flow" (2019).</summary>
/// <remarks>This cost function attempts to achieve collision avoidance based on a neighbor's focus of expansion (FOE), 
/// which is a point in an agent's view indicating the neighbor's relative motion.
///
/// The gradient of this cost function has been explicitly implemented.
/// Thus, this cost function is suitable for use in a gradient-based policy.
/// 
/// ### Name in XML files:
/// <tt>%FOEAvoidance</tt>
/// ### Parameters:
/// None
/// </remarks>
class FOEAvoidance : public CostFunction
{
 public:
	FOEAvoidance() : CostFunction() {}
	virtual ~FOEAvoidance() {}
	const static std::string GetName() { return "FOEAvoidance"; }
  
	virtual float GetCost(const Vector2D& velocity, Agent* agent, const WorldBase * world) const override;
	virtual Vector2D GetGradient(const Vector2D& velocity, Agent* agent, const WorldBase * world) const override;
};

#endif //LIB_FOE_AVOIDANCE_H
