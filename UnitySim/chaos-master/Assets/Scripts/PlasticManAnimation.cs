/* Crowd Simulator Engine
** Copyright (C) 2018 - Inria Rennes - Rainbow - Julien Pettre
**
** This program is free software; you can redistribute it and/or
** modify it under the terms of the GNU General Public License
** as published by the Free Software Foundation; either version 2
** of the License, or (at your option) any later version.
**
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with this program; if not, write to the Free Software
** Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
**
** Authors: Julien Bruneau
**
** Contact: crowd_group@inria.fr
*/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Control the walking animation 
/// </summary>
public class PlasticManAnimation : MonoBehaviour {

    private Vector3 _oldPosition;       // previous position
    private Animator _objectAnimator;   // Animator

    /// <summary>
    /// Init the animation
    /// </summary>
    void Start ()
    {
        _oldPosition = this.transform.position;
        _objectAnimator = this.GetComponent<Animator>();
        _objectAnimator.applyRootMotion = false;

        _objectAnimator.SetFloat("CycleOffset", Random.value);
        _objectAnimator.speed = 0;
        _objectAnimator.SetBool("IsIddle", true);
    }

    /// <summary>
    /// Update animation parameters to follow the agent's trajectory
    /// </summary>
    void Update () {
        if (Time.deltaTime != 0)
        {
            //If the time isn't in pause, play the animation as fast as the virtual human speed.
            float animationSpeed;
            Vector3 position = _oldPosition - this.transform.position;
            animationSpeed = position.magnitude / Time.deltaTime / 1.4f;

            if (animationSpeed == 0)
            {
                _objectAnimator.speed = 1;
                _objectAnimator.SetBool("IsIddle", true);
            }
            else
            {
                _objectAnimator.speed = animationSpeed;
                _objectAnimator.SetBool("IsIddle", false);
            }

        }
        else
        {
            // if the time is in pause, pause the animation.
            _objectAnimator.speed = 0;
        }

        _oldPosition = this.transform.position;
    }
}
