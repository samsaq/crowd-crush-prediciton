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

#ifndef LIB_KARAMOUZAS_H
#define LIB_KARAMOUZAS_H

#include <core/costFunction.h>

/// @ingroup costfunctions
/// <summary>An implementation of the collision-avoidance cost function proposed by %Karamouzas & Overmars, 
/// in the paper "A velocity-based approach for simulating human collision avoidance" (2010).</summary>
/// <remarks>
/// ### Notes: 
/// <list>
/// <item>This cost function already includes a goal-reaching component.</item>
/// <item>The <tt>%Karamouzas.xml</tt> policy in our example files reproduces the original algorithm.</item>
/// </list>
/// 
/// ### Name in XML files:
/// <tt>%Karamouzas</tt>
///
/// ### Parameters: 
/// <list>
/// <item>alpha, beta, gamma, delta (float): Scalars for the four main components of the cost function.</item>
/// <item>t_max (float): The maximum time-to-collision to consider.</item>
/// </list>
/// </remarks>
class Karamouzas : public CostFunction
{
private:
	float alpha = 5;
	float beta = 0.5f;
	float gamma = 1;
	float delta = 1;
	float t_max = 8;

	// thresholds for time to collision
	const float tc_min = 2.5f;
	const float tc_mid = 6;
	const float tc_max = t_max;

	// thresholds for the max angular deviation
	const float d_min = 0.05f; // theoretically 0, but we don't want to accidentally ignore the preferred velocity due to numerical errors
	const float d_mid = (float)(PI / 6.0f);
	const float d_max = (float)(PI / 2.0f);

public:
	Karamouzas() : CostFunction() {}
	virtual ~Karamouzas() {}
	const static std::string GetName() { return "Karamouzas"; }

	virtual float GetCost(const Vector2D& velocity, Agent* agent, const WorldBase * world) const override;
	void parseParameters(const CostFunctionParameters & params) override;

private:
	float getMaxDeviationAngle(const Agent* agent, const float ttc) const;
	float getMinSpeed(const Agent* agent, const float ttc) const;
	float getMaxSpeed(const Agent* agent, const float ttc) const;
};

#endif //LIB_KARAMOUZAS_H
