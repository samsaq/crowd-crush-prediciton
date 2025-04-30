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

#include <core/agent.h>
#include <core/worldBase.h>

Agent::Agent(size_t id, const Agent::Settings& settings) :
	id_(id), settings_(settings),
	position_(0, 0),
	velocity_(0, 0),
	acceleration_(0, 0),
	contact_forces_(0, 0),
	preferred_velocity_(0, 0),
	goal_(0, 0),
	viewing_direction_(0, 0),
	next_acceleration_(0, 0),
	next_contact_forces_(0, 0)
{
	// set the seed for random-number generation
	RNGengine_.seed((unsigned int)id);
}

#pragma region [Simulation-loop methods]

void Agent::ComputeNeighbors(WorldBase* world)
{
	// get the query radius
	float range = getPolicy()->getInteractionRange();

	// perform the query and store the result
	neighbors_ = world->ComputeNeighbors(position_, range, this);
}

void Agent::ComputePreferredVelocity()
{
	if (hasReachedGoal())
		preferred_velocity_ = Vector2D(0, 0);

	else
		preferred_velocity_ = (goal_ - position_).getnormalized() * getPreferredSpeed();
}

void Agent::ComputeAcceleration(WorldBase* world)
{
	next_acceleration_ = getPolicy()->ComputeAcceleration(this, world);
}

void Agent::ComputeContactForces(WorldBase* world)
{
	next_contact_forces_ = getPolicy()->ComputeContactForces(this, world);
}

void Agent::updateViewingDirection()
{
	// update the viewing direction:
	// weighted average of the preferred and actual velocity
	const auto& vDiff = (2 * velocity_ + preferred_velocity_) / 3;
	if (vDiff.sqrMagnitude() > 0.01)
		viewing_direction_ = vDiff.getnormalized();
}

void Agent::UpdateVelocityAndPosition(WorldBase* world)
{
	const float dt = world->GetDeltaTime();

	// clamp the acceleration
	acceleration_ = clampVector(next_acceleration_, getMaximumAcceleration());

	// integrate the velocity; clamp to a maximum speed
	velocity_ = clampVector(velocity_ + (acceleration_*dt), getMaximumSpeed());

	// add contact forces
	contact_forces_ = next_contact_forces_;
	velocity_ += contact_forces_ / settings_.mass_ * dt;

	// update the position
	position_ += velocity_ * dt;

	updateViewingDirection();
}

#pragma endregion

#pragma region [Advanced getters]

bool Agent::hasReachedGoal() const
{
	return (goal_ - position_).sqrMagnitude() <= getRadius() * getRadius();
}

#pragma endregion

#pragma region [Basic setters]

void Agent::setPosition(const Vector2D &position)
{
	position_ = position;
}

void Agent::setVelocity_ExternalApplication(const Vector2D &velocity, const Vector2D &viewingDirection)
{
	velocity_ = velocity;
	viewing_direction_ = viewingDirection;
}

void Agent::setGoal(const Vector2D &goal)
{
	goal_ = goal;

	// look straight towards the goal
	if (goal_ != position_)
		viewing_direction_ = (goal_ - position_).getnormalized();
}

#pragma endregion

float Agent::ComputeRandomNumber(float min, float max)
{
	auto distribution = std::uniform_real_distribution<float>(min, max);
	return distribution(RNGengine_);
}