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

public class GetMemberPositionPlasticMan : GetMemberPosition
{
    protected override void InitList()
    {
        BodyListMember = new List<string>
        {
            "Plasticman:LeftWrist_End",
            "Plasticman:RightWrist_End",
            "Plasticman:LeftForeArm",
            "Plasticman:RightForeArm",
            "Plasticman:LeftFoot",
            "Plasticman:RightFoot",
            "PPlasticman:RFoot_End",
            "Plasticman:LFoot_End",
            "Plasticman:LeftLeg",
            "Plasticman:RightLeg",
            "Plasticman:Head_End"
        };

        HeadListMember = new List<string>
        {
            // Head
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
