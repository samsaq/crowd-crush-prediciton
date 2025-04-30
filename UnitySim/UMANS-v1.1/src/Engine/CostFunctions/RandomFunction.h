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

#ifndef LIB_RANDOM_FUNCTION_H
#define LIB_RANDOM_FUNCTION_H

#include <core/costFunction.h>

/// @ingroup costfunctions
/// <summary>A "fake" navigation function that produces random costs and gradients.</summary>
/// <remarks>This cost function can be used to add noise to an agent's navigation.
/// ### Name in XML files:
/// <tt>%RandomFunction</tt>
/// ### Parameters: 
/// None. You can use the common <tt>coeff</tt> parameter to scale the effect of this cost function.
/// </remarks>
class RandomFunction : public CostFunction
{
public:
	RandomFunction() : CostFunction() { range_ = 0; }
	virtual ~RandomFunction() {}
	const static std::string GetName() { return "RandomFunction"; }

	/// Computes a random cost between -1 and 1.
	virtual float GetCost(const Vector2D& velocity, Agent* agent, const WorldBase * world) const override;

	/// Computes a random gradient with both the X and Y component between -1 and 1.
	virtual Vector2D GetGradient(const Vector2D& velocity, Agent* agent, const WorldBase * world) const override;
};

#endif //LIB_RANDOM_FUNCTION_H
