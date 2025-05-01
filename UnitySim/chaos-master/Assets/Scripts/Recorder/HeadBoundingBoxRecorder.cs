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

using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

public class HeadBoundingBoxRecorder : DeepRecorder
{
    /// <summary>
    /// This function will create the output data file for body bounding boxes
    /// </summary>
    public override void Init(string Directory)
    {
        base.Init(Directory);
        FileData = _SaveDir + "HeadBoundingBoxes.csv";
        StreamWriter writer = new StreamWriter(FileData, false);
        writer.Close();
    }

    /// <summary>
    /// This function will collect the body bounding boxes need for deep learning
    /// <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    /// </summary>
    /// <returns></returns>
    protected override string GetDeepData(int IdFrame)
    {
        Camera Cam = Camera.main;
        string text = "";
        foreach (GameObject human in GameObject.FindGameObjectsWithTag("Player"))
        {
            if (human.GetComponentInChildren<SkinnedMeshRenderer>().enabled)
            {
                var component = human.GetComponent<GetMemberPosition>();
                if (component != null)
                {
                    var bbox = component.GetHeadBoundingBox();

                    // check bounding boxes
                    if (bbox[2] > bbox[0] && bbox[3] > bbox[1])
                    {
                        text += IdFrame.ToString() + "," +
                                component.ID.ToString() + "," +
                                bbox[0].ToString().Replace(',', '.') + "," +
                                (Cam.pixelHeight - bbox[1]).ToString().Replace(',', '.') + "," +
                                bbox[2].ToString().Replace(',', '.') + "," +
                                (Cam.pixelHeight - bbox[3]).ToString().Replace(',', '.') + "," +
                                "-1,-1,-1,-1\n";
                    }
                }
            }
        }
        return text;
    }
}