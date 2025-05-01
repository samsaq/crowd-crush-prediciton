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
public abstract class AFilter
{
	// variables
	bool _UseOnline;
	TrajectoryReader _Reader;
	int _IndexCount;
	
	public TrajectoryReader Reader
	{
		get
		{
			return _Reader;
		}
		
		set
		{
			_Reader = value;
		}
	}

	public int IndexCount
	{
		get
		{
			return _IndexCount;
		}
		
		set
		{
			_IndexCount = value;
		}
	}
	
	// methods
	abstract public void SmootRotation(float fCurrentTime, int currentIndex, out Vector3 LocalTargetInfOnTrajectory, out Vector3 LocalTargetSupOnTrajectory);

	abstract public void SmootPosition(float fCurrentTime, int currentIndex, out Vector3 LocalTargetInfOnTrajectory, out Vector3 LocalTargetSupOnTrajectory);

	public Vector3 GetPositionAtTime(float timerequested, bool increment, int currentIndex)
	{
		if(timerequested<_Reader.IndexList[0][1])
		{
            float y=0.0f;
            //if (_Reader.mSkeleton[0][1]<0 || _Reader.mSkeleton[0][1]>50 || _Reader.mSkeleton[0][2]<-25 || _Reader.mSkeleton[0][2]>25)
            //    y=-1000.0f;

			return new Vector3(_Reader.mSkeleton[0][1], y, _Reader.mSkeleton[0][2]);
		}
		else if(timerequested>_Reader.IndexList[_Reader.mSkeleton.Count-1][1])
		{
            float y=0.0f;
            //if (_Reader.mSkeleton[_Reader.mSkeleton.Count-1][1]<0 || _Reader.mSkeleton[_Reader.mSkeleton.Count-1][1]>50 || _Reader.mSkeleton[_Reader.mSkeleton.Count-1][2]<-25 || _Reader.mSkeleton[_Reader.mSkeleton.Count-1][2]>25)
            //    y=-1000.0f;
			return new Vector3(_Reader.mSkeleton[_Reader.mSkeleton.Count-1][1], y, _Reader.mSkeleton[_Reader.mSkeleton.Count-1][2]);
		}
		else
		{
			Vector4 vInfTemp = new Vector4();
			Vector4 vSupTemp = new Vector4();
			bool ok = FindIntervalFromTime(timerequested, ref vInfTemp, ref vSupTemp, increment, currentIndex);
			if(!ok)
			{
				return new Vector3();
			}
			float TVInf = vInfTemp[0];
			float TVSup = vSupTemp[0];
			float TReference = TVSup-TVInf;
			float TInInterval = timerequested-TVInf;
			float RatioTimeInInterval = TInInterval/TReference;

            float y = 0.0f;
            //if (vInfTemp[1] < 0 || vInfTemp[1] > 50 || vInfTemp[2] < -25 || vInfTemp[2] > 25)
            //    y = -1000.0f;
            //if (vSupTemp[1] < 0 || vSupTemp[1] > 50 || vSupTemp[2] < -25 || vSupTemp[2] > 25)
            //    y = -1000.0f;
            Vector3 tmpvInfInWorld = new Vector3(vInfTemp[1], y, vInfTemp[2]);
			Vector3 tmpvSupInWorld = new Vector3(vSupTemp[1], y, vSupTemp[2]);
			Vector3 DirectionOfTheInterval = tmpvSupInWorld - tmpvInfInWorld;
			Vector3 CurrentPositionOnTheVector = tmpvInfInWorld+RatioTimeInInterval*DirectionOfTheInterval;
			return CurrentPositionOnTheVector;
		}
	}

	bool FindIntervalFromTime(float time, ref Vector4 vInfTemp, ref Vector4 vSupTemp, bool increment, int currentIndex)
	{
		if(currentIndex>=_IndexCount)
		{
			return false;
		}
		if(increment)
		{
			// test en avancant dans la liste 
			for(int iIndex = currentIndex ; iIndex<_IndexCount ; ++iIndex)
			{
				float fCurrentIndexTime = 0.0f;
				float fNextIndexTime = 0.0f;
				
				if(iIndex+1<_IndexCount)
				{
					fCurrentIndexTime = _Reader.IndexList[iIndex][1];
					fNextIndexTime = _Reader.IndexList[iIndex+1][1];
					
					if( (time>=fCurrentIndexTime) && (time<fNextIndexTime))
					{					
						vInfTemp = _Reader.mSkeleton[iIndex];
						vSupTemp = _Reader.mSkeleton[iIndex+1];
						return true;
					}
				}
			}
		}
		else
		{
			// si pas de solution trouvée, test en reculant dans la liste 
			for(int iIndex = currentIndex ; iIndex>=0 ; --iIndex)
			{
				float fCurrentIndexTime = 0.0f;
				float fNextIndexTime = 0.0f;
				
				if(iIndex+1<_IndexCount)
				{
					fCurrentIndexTime = _Reader.IndexList[iIndex][1];
					fNextIndexTime = _Reader.IndexList[iIndex+1][1];
					
					if( (time>=fCurrentIndexTime) && (time<fNextIndexTime))
					{					
						vInfTemp = _Reader.mSkeleton[iIndex];
						vSupTemp = _Reader.mSkeleton[iIndex+1];
						return true;
					}
				}
			}
		}
		return false;
	}

	public TrajectoryReader getReader()
	{
		return _Reader;
	}
}