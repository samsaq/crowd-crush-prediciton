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
** Authors: Tristan Le Bouffant, Julian Joseph
**
** Contact: crowd_group@inria.fr
*/

using UnityEngine;

[System.Serializable]
public class BasicFilter : AFilter
{
	public float _DeltaTimeForSmoothing;
	public bool _SmoothingRotation = false;
	public bool _SmoothingPosition = false;


	public BasicFilter()
	{
	}

	public override void SmootRotation(float fCurrentTime, int currentIndex, out Vector3 LocalTargetInfOnTrajectory, out Vector3 LocalTargetSupOnTrajectory)
	{
		LocalTargetInfOnTrajectory = GetPositionAtTime(fCurrentTime-_DeltaTimeForSmoothing, false, currentIndex);
		LocalTargetSupOnTrajectory = GetPositionAtTime(fCurrentTime+_DeltaTimeForSmoothing, true, currentIndex);
	}
	
	public override void SmootPosition(float fCurrentTime, int currentIndex, out Vector3 LocalTargetInfOnTrajectory, out Vector3 LocalTargetSupOnTrajectory)
	{
		LocalTargetInfOnTrajectory = GetPositionAtTime(fCurrentTime-_DeltaTimeForSmoothing, false, currentIndex);
		LocalTargetSupOnTrajectory = GetPositionAtTime(fCurrentTime+_DeltaTimeForSmoothing, true, currentIndex);
	}

}

