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
using System.Collections.Generic;

public class GetMemberPositionRocketBox : GetMemberPosition
{
    protected override void InitList()
    {
        BodyListMember = new List<string>
        {
            // Head
            "Bip01 HeadNub",
            "Bip01 Neck",
            "Bip01 MNoseNub",
            "Bip01 LMasseterNub",
            "Bip01 RMasseterNub",

            // Left arm
            "Bip01 L Clavicle",
            "Bip01 L UpperArm",
            "Bip01 L Forearm",
            "Bip01 L Hand",
            "Bip01 L Finger2Nub",

            // Right arm
            "Bip01 R Clavicle",
            "Bip01 R UpperArm",
            "Bip01 R Forearm",
            "Bip01 R Hand",
            "Bip01 R Finger2Nub",

            //Left Leg
            "Bip01 L Thigh",
            "Bip01 L Calf",
            "Bip01 L Foot",
            "Bip01 L Toe0Nub",
            
            //Right Leg
            "Bip01 R Thigh",
            "Bip01 R Calf",
            "Bip01 R Foot",
            "Bip01 R Toe0Nub"
        };

        HeadListMember = new List<string>
        {
            "Hbb_TopHead",
            "Hbb_BackHead",
            "Hbb_Nose",
            "Hbb_Chin",
            "Hbb_L_Ear",
            "Hbb_R_Ear",
            "Hbb_BackBottomHead"
        };
    }
}
