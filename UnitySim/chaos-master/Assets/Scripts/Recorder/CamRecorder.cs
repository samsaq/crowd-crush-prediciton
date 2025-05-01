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
** Authors: Julien Bruneau, Tristan Le Bouffant
**
** Contact: crowd_group@inria.fr
*/
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// Control camera recording by taking screenshot at specific framerate
/// </summary>
public class CamRecorder : MonoBehaviour
{

    #region attributes
    protected int imageIncrement = 0;                 // NB images already save for incrementing files name
    public bool record = true;                      // Is recording or not

    public float timeToStart = 0;                   // Time when recording shall start
    public float timeToStop = 60;                   // Time when recording shall stop
    public int framerate = 25;                      // Framerate at which screenshot are taken
    public string saveDir = "Img/capture/";         // Directory where to save all the data

    private bool runOncePerFrame = false;
    #endregion

    /// <summary>
    /// Initialize the recording
    /// </summary>
    protected virtual void Start()
    {
        StartCoroutine(RecordDataScreen());
    }

    /// <summary>
    /// Change the recording framerate
    /// </summary>
    /// <param name="rate">new framerate</param>
    public void ChangeFramerate(int rate)
    {
        framerate = rate;
    }

    /// <summary>
    /// initialize constant and create output directory for images
    /// </summary>
    /// <param name="rate">new framerate</param>
    public virtual void Init()
    {
        Directory.CreateDirectory(saveDir + "Images");
        imageIncrement = 0;
    }

    private void Update()
    {
        runOncePerFrame = true;
    }

    /// <summary>
    /// Create screenshot during recording time
    /// </summary>
    IEnumerator RecordDataScreen()
    {
        // We should only read the screen buffer after rendering is complete
        yield return new WaitForEndOfFrame();

        if (record && !(Time.timeSinceLevelLoad < timeToStart))
        {
            if (Time.captureFramerate == 0)
                Time.captureFramerate = framerate;

            if (Time.timeSinceLevelLoad > timeToStop)
            {
                record = false;
                Time.captureFramerate = 0;
                Debug.Log("record stopped !");
                Application.Quit();
            }
            else if (Time.timeScale != 0)
            {

                //ScreenCapture.CaptureScreenshot(saveDir + "/" + imageIncrement.ToString("D" + 4) + ".png");

                this.GetComponent<FiltersControl>().Save(imageIncrement.ToString("D" + 4) + ".png", ConfigReader.recordingWidth, ConfigReader.recordingHeight, saveDir);

                // Record data need for deep learning
                foreach (DeepRecorder rec in gameObject.GetComponentsInChildren<DeepRecorder>())
                {
                    rec.RecordDeepData(imageIncrement);
                }

                imageIncrement++;
            }
        }
        else
            Time.captureFramerate = 0;

        StartCoroutine(RecordDataScreen());
    }

}
