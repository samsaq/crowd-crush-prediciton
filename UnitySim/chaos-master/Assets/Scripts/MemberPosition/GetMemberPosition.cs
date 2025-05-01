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
** Authors: Florian Berton, Wouter Van Toll
**
** Contact: crowd_group@inria.fr
*/

using System;
using System.Collections.Generic;
using UnityEngine;

public abstract class GetMemberPosition : MonoBehaviour {

    // Names of child objects to use for bounding-box computations
    protected List<string> BodyListMember;
    protected List<string> HeadListMember;

    bool IncludePointsOutsideCamera = false;
    int NbPointsInsideCam = 0;
    
    // Id of the virtual human
    public int ID { get; set; }
    // Treshold for number of points inside camera
    int ThPointsInsideCam = 5;

	// Use this for initialization
	void Start ()
    {
        InitList();
    }

    protected abstract void InitList();

    void UpdateBoundingBoxRecursive(List<string> ListMember, Transform transform, List<double> Box)
    {
        if (ListMember.Contains(transform.name) && !IsPointBehindCamera(transform.position))
        {
            Vector3 PosC = Camera.main.WorldToScreenPoint(transform.position);
            if (IsPointInsideCamera(PosC))
                NbPointsInsideCam++;

            Box[0] = Math.Min(Box[0], PosC.x);
            Box[1] = Math.Min(Box[1], PosC.y);
            Box[2] = Math.Max(Box[2], PosC.x);
            Box[3] = Math.Max(Box[3], PosC.y);

        }

        foreach (Transform child in transform)
            UpdateBoundingBoxRecursive(ListMember,child, Box);
    }

    private bool IsPointBehindCamera(Vector3 posW)
    {
        return Vector3.Dot(Camera.main.transform.forward, posW - Camera.main.transform.position) < 0;
    }

    private bool IsPointInsideCamera(Vector3 PosC)
    {
        Camera Cam = Camera.main;
        Rect screen = new Rect(0, 0, Cam.pixelWidth, Cam.pixelWidth);
        return screen.Contains(PosC);
    }

    public List<double> GetBoundingBox(List<string> ListMember)
    {
        List<double> res = new List<double> { Camera.main.pixelWidth, Camera.main.pixelHeight,0,0};
        NbPointsInsideCam = 0; // Reset the number of element outside of the camera

        // get the bounding box of all relevant points in screen coordinates
        UpdateBoundingBoxRecursive(ListMember,transform, res);

        // check how many points were inside the camera
        if (NbPointsInsideCam < ThPointsInsideCam)
        {
            return new List<double> { Camera.main.pixelWidth, Camera.main.pixelHeight, 0, 0 };
        }
        else
        {
            return res;
        }
    }

    //summary function to get boundingboxes for the whole body
    public List<double> GetBodyBoundingBox()
    {
        return this.GetBoundingBox(this.BodyListMember);
    }

    //summary function to get boundingboxes for the gead
    public List<double> GetHeadBoundingBox()
    {
        return this.GetBoundingBox(this.HeadListMember);
    }
}
