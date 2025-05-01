﻿/* Crowd Simulator Engine
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

public abstract class DeepRecorder : MonoBehaviour
{
    protected GameObject[] HumanList;                 // List of Virtual Human
    protected string FileData;                        // File with the data
    protected string _SaveDir;                        // Directory where to save all the data

    public virtual void Init(string Directory)
    {
        _SaveDir = Directory;
    }

    /// <summary>
    /// This function will write the data need for deep learning
    public void RecordDeepData(int IdFrame)
    {
        string data = GetDeepData(IdFrame);
        File.AppendAllText(FileData, data);
    }

    /// <summary>
    /// This function will collect the data need for deep learning
    protected abstract string GetDeepData(int IdFrame);

}
