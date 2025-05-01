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
** Authors: Fabrice Atrevi
**
** Contact: crowd_group@inria.fr
*/

using UnityEngine;
using System.IO;
using System.Xml;
using System.Xml.Serialization;
using System.Text;
using UnityEngine.UI;
using System.Collections.Generic;
using SFB;

/// <summary>
/// Static class used to set the simulation parameters from the interface.
/// </summary>
public class ConfigManager : MonoBehaviour
{
    // GameObject to store differents panel in the canvas
    public static GameObject PanelCamera, PanelRecording, PanelTimeSetting, PanelStageEnv;

    public static InputField LookAgent, FollowAgent;

    /// <summary>
    /// Method to show on the panel the parameters from the config file.
    /// </summary>
    public static void showConfig()
    {
        /// Get Camera Config
        //Type of Camera
        PanelCamera = GameObject.Find("/ConfigMenu/Panel/PanelCamera/Button/Panel");
        Dropdown camSelect = PanelCamera.transform.Find("cameraType").GetComponent<Dropdown>();
        camSelect.value = ConfigReader.camType;

        // Position
        Vector3 camPos = ConfigReader.camPosition;
        PanelCamera.transform.Find("PosX").GetComponent<InputField>().text = camPos.x.ToString();
        PanelCamera.transform.Find("PosY").GetComponent<InputField>().text = camPos.y.ToString();
        PanelCamera.transform.Find("PosZ").GetComponent<InputField>().text = camPos.z.ToString();
        
        // Rotation
        Vector3 camRot = ConfigReader.camRotation;
        PanelCamera.transform.Find("RotX").GetComponent<InputField>().text = camRot.x.ToString();
        PanelCamera.transform.Find("RotY").GetComponent<InputField>().text = camRot.y.ToString();
        PanelCamera.transform.Find("RotZ").GetComponent<InputField>().text = camRot.z.ToString();
        
        // Agent ID
        if(camSelect.options[camSelect.value].text == "Look_At")
            PanelCamera.transform.Find("AgentID").GetComponent<InputField>().text = ConfigReader.camLookAtTarget.ToString();
        else if(camSelect.options[camSelect.value].text == "Follow" || camSelect.options[camSelect.value].text == "First_Person" || camSelect.options[camSelect.value].text == "Torsum")
            PanelCamera.transform.Find("AgentID").GetComponent<InputField>().text = ConfigReader.camFollowTarget.ToString();

        // Follow Agent
        PanelCamera.transform.Find("FollowX").GetComponent<Toggle>().isOn = ConfigReader.camFollowOnX;
        PanelCamera.transform.Find("FollowY").GetComponent<Toggle>().isOn = ConfigReader.camFollowOnY;
        PanelCamera.transform.Find("LockPerson").GetComponent<Toggle>().isOn = ConfigReader.camLockFirstPerson;
        PanelCamera.transform.Find("SmoothView").GetComponent<Toggle>().isOn = ConfigReader.camSmoothFirstPerson;

        /// Get Recording parameters
        // Duration
        PanelRecording   = GameObject.Find("/ConfigMenu/Panel/PanelRecording/Button/Panel");
        PanelTimeSetting = GameObject.Find("/ConfigMenu/Panel/PanelRecording/Button/Panel/timeSettingPanel");
        PanelRecording.transform.Find("recordVal").GetComponent<Toggle>().isOn  = ConfigReader.recording;
        PanelTimeSetting.transform.Find("startVal").GetComponent<InputField>().text = ConfigReader.recordingStart.ToString();
        PanelTimeSetting.transform.Find("endVal").GetComponent<InputField>().text = ConfigReader.recordingEnd.ToString();

        // FPS
        PanelTimeSetting.transform.Find("fps").GetComponent<InputField>().text = ConfigReader.recordingFramerate.ToString();
        
        // Resolution
        PanelRecording.transform.Find("W").GetComponent<InputField>().text = ConfigReader.recordingWidth.ToString();
        PanelRecording.transform.Find("H").GetComponent<InputField>().text = ConfigReader.recordingHeight.ToString();

        // Save Path
        PanelRecording.transform.Find("savedirVal").GetComponent<Text>().text = ConfigReader.recordingSaveDir;

        // Save Datas
        PanelRecording.transform.Find("imgOri").GetComponent<Toggle>().isOn    = ConfigReader.imgOriginal.record;
        PanelRecording.transform.Find("imgSeg").GetComponent<Toggle>().isOn    = ConfigReader.imgSegmentation.record;
        PanelRecording.transform.Find("imgCat").GetComponent<Toggle>().isOn    = ConfigReader.imgCategories.record;
        PanelRecording.transform.Find("imgDepth").GetComponent<Toggle>().isOn  = ConfigReader.imgDepth.record;
        PanelRecording.transform.Find("imgNor").GetComponent<Toggle>().isOn    = ConfigReader.imgNormals.record;
        PanelRecording.transform.Find("imgOF").GetComponent<Toggle>().isOn     = ConfigReader.imgOpticalFlow.record;
        PanelRecording.transform.Find("imgBBox").GetComponent<Toggle>().isOn   = ConfigReader.bboxeBody.record;
        PanelRecording.transform.Find("imgHBox").GetComponent<Toggle>().isOn   = ConfigReader.bboxeHead.record;

        /// Get Environnement parameters
        PanelStageEnv = GameObject.Find("/ConfigMenu/Panel/PanelStageEnv/Button/Panel");
        PanelStageEnv.transform.Find("EnvFilePathVal").GetComponent<Text>().text = ConfigReader.trajectoriesDir;
        PanelStageEnv.transform.Find("ObstFilePathVal").GetComponent<Text>().text = ConfigReader.obstaclesFile;
    }

    /// <summary>
    /// Method to set the config file using information from the interface.
    /// </summary>
    /// <param name="configPath"> path to the config file</param>
    public static void SaveConfig(string configPath)
    {
        //// Set Camera Config
        // Type of camera
        PanelCamera = GameObject.Find("/ConfigMenu/Panel/PanelCamera/Button/Panel");
        Dropdown camSelect = PanelCamera.transform.Find("cameraType").GetComponent<Dropdown>();
        ConfigReader.camType = camSelect.value;
        
        // Position
        Vector3 camPos;
        camPos.x = float.Parse(PanelCamera.transform.Find("PosX").GetComponent<InputField>().text);
        camPos.y = float.Parse(PanelCamera.transform.Find("PosY").GetComponent<InputField>().text);
        camPos.z = float.Parse(PanelCamera.transform.Find("PosZ").GetComponent<InputField>().text);

        ConfigReader.camPosition = camPos;

        // Rotation
        Vector3 camRot;
        camRot.x = float.Parse(PanelCamera.transform.Find("RotX").GetComponent<InputField>().text);
        camRot.y = float.Parse(PanelCamera.transform.Find("RotY").GetComponent<InputField>().text);
        camRot.z = float.Parse(PanelCamera.transform.Find("RotZ").GetComponent<InputField>().text);
        
        ConfigReader.camRotation = camRot;

        // Look At Agent
        if(camSelect.options[camSelect.value].text == "Look_At")
        {
            ConfigReader.camLookAtTarget = int.Parse(PanelCamera.transform.Find("AgentID").GetComponent<InputField>().text);
        }
        else{
            ConfigReader.camLookAtTarget = -1;
        }

        // Follow Agent
        if(camSelect.options[camSelect.value].text == "Follow"){
            ConfigReader.camFollowTarget = int.Parse(PanelCamera.transform.Find("AgentID").GetComponent<InputField>().text);
            ConfigReader.camFollowOnX = PanelCamera.transform.Find("FollowX").GetComponent<Toggle>().isOn;
            ConfigReader.camFollowOnY = PanelCamera.transform.Find("FollowY").GetComponent<Toggle>().isOn;
        }
        else{
            ConfigReader.camFollowTarget = -1;
        }

        // First Person and Torsum
        if(camSelect.options[camSelect.value].text == "First_Person" || camSelect.options[camSelect.value].text == "Torsum")
        {
            ConfigReader.camFollowTarget = int.Parse(PanelCamera.transform.Find("AgentID").GetComponent<InputField>().text);
            ConfigReader.camLockFirstPerson = PanelCamera.transform.Find("LockPerson").GetComponent<Toggle>().isOn;
            ConfigReader.camSmoothFirstPerson = PanelCamera.transform.Find("SmoothView").GetComponent<Toggle>().isOn;
        }

        //// Set Recording parameters
        // Duration
        PanelRecording = GameObject.Find("/ConfigMenu/Panel/PanelRecording/Button/Panel");
        PanelTimeSetting = GameObject.Find("/ConfigMenu/Panel/PanelRecording/Button/Panel/timeSettingPanel");

        if(PanelRecording.transform.Find("recordVal").GetComponent<Toggle>().isOn)
        {
            ConfigReader.recordingStart = int.Parse(PanelTimeSetting.transform.Find("startVal").GetComponent<InputField>().text);
            ConfigReader.recordingEnd = int.Parse(PanelTimeSetting.transform.Find("endVal").GetComponent<InputField>().text);

            // FPS
            ConfigReader.recordingFramerate = int.Parse(PanelTimeSetting.transform.Find("fps").GetComponent<InputField>().text);
        }
        else{
            ConfigReader.recordingStart = 0;
            ConfigReader.recordingEnd   = 0;
        }

        // Resolution
        ConfigReader.recordingWidth = int.Parse(PanelRecording.transform.Find("W").GetComponent<InputField>().text);
        ConfigReader.recordingHeight = int.Parse(PanelRecording.transform.Find("H").GetComponent<InputField>().text);

        // Save Datas
        ConfigReader.imgOriginal.record = PanelRecording.transform.Find("imgOri").GetComponent<Toggle>().isOn;
        ConfigReader.imgSegmentation.record = PanelRecording.transform.Find("imgSeg").GetComponent<Toggle>().isOn;
        ConfigReader.imgCategories.record = PanelRecording.transform.Find("imgCat").GetComponent<Toggle>().isOn;
        ConfigReader.imgDepth.record = PanelRecording.transform.Find("imgDepth").GetComponent<Toggle>().isOn;
        ConfigReader.imgNormals.record = PanelRecording.transform.Find("imgNor").GetComponent<Toggle>().isOn;
        ConfigReader.imgOpticalFlow.record = PanelRecording.transform.Find("imgOF").GetComponent<Toggle>().isOn;
        ConfigReader.bboxeBody.record = PanelRecording.transform.Find("imgBBox").GetComponent<Toggle>().isOn;
        ConfigReader.bboxeHead.record = PanelRecording.transform.Find("imgHBox").GetComponent<Toggle>().isOn;
        
        ConfigReader.SaveConfig(configPath);
    }
        
    /// <summary>
    /// Method to select the save directory.
    /// </summary>
    public void saveDirBrowse()
    {
        // Save Path
        string oldPath, newPath;

        PanelRecording = GameObject.Find("/ConfigMenu/Panel/PanelRecording/Button/Panel");

        oldPath = PanelRecording.transform.Find("savedirVal").GetComponent<Text>().text;
        newPath = OpenFolderExplorer();

        if(newPath == "")
        {
            ConfigReader.trajectoriesDir = oldPath;
        }
        else
        {
            newPath = newPath+"/";
            ConfigReader.recordingSaveDir = newPath;
            PanelRecording.transform.Find("savedirVal").GetComponent<Text>().text = newPath;
        }
    }

    /// <summary>
    /// Method to select the environnement directory
    /// </summary>
    public void envDirBrowse()
    {
        // Save Path
        string oldPath, newPath;

        PanelStageEnv = GameObject.Find("/ConfigMenu/Panel/PanelStageEnv/Button/Panel");

        oldPath = PanelStageEnv.transform.Find("EnvFilePathVal").GetComponent<Text>().text;
        newPath = OpenFolderExplorer();

        if(newPath == "")
        {
            ConfigReader.recordingSaveDir = oldPath;
        }
        else
        {
            newPath = newPath+"/";
            ConfigReader.trajectoriesDir = newPath;
            PanelStageEnv.transform.Find("EnvFilePathVal").GetComponent<Text>().text = newPath;
        }
    }
        
    /// <summary>
    /// Method to select the obstacle definition file.
    /// </summary>
    public void obstacleFileBrowse()
    {
        // File Path
        string oldPath, newPath;

        PanelStageEnv = GameObject.Find("/ConfigMenu/Panel/PanelStageEnv/Button/Panel");

        oldPath = PanelStageEnv.transform.Find("ObstFilePathVal").GetComponent<Text>().text;
        newPath = OpenFileExplorer("xml");

        if(newPath == "")
        {
            ConfigReader.obstaclesFile = oldPath;
        }
        else
        {
            ConfigReader.obstaclesFile = newPath;
            PanelStageEnv.transform.Find("ObstFilePathVal").GetComponent<Text>().text = newPath;
        }
    }

    /// <summary>
    /// Method to open the browser to select a folder.
    /// </summary>
    /// <returns> selected folder's path</returns>
    public string OpenFolderExplorer()
    {
        var folderPath = StandaloneFileBrowser.OpenFolderPanel("Choose a folder", "", false);
    
        string path="";
        foreach (var p in folderPath) {
            path += p;
        }
        return path;
    }

    /// <summary>
    /// Method to open the browser to select a file.
    /// </summary>
    /// <param name="extension"> filter the files in the folder by the extension </param>
    /// <returns> selected file's path</returns>
    public string OpenFileExplorer(string extension)
    {
        string[] filePath = StandaloneFileBrowser.OpenFilePanel("Choose a File", "", extension, false);
    
        string path="";
        foreach (var p in filePath) {
            path += p;
        }
        return path;
    }

    /// <summary>
    /// Method to check if the record toggle is on or off and display the time setting parameters accordingly.
    /// </summary>
    public void checkRecordToggle(){
        PanelRecording = GameObject.Find("/ConfigMenu/Panel/PanelRecording/Button/Panel");
        PanelTimeSetting = GameObject.Find("/ConfigMenu/Panel/PanelRecording/Button/Panel/timeSettingPanel");

        PanelTimeSetting.SetActive(PanelRecording.transform.Find("recordVal").GetComponent<Toggle>().isOn);
    }

    /// <summary>
    /// Method to check if the lookAt and FollowAt toggle are on or off and display the suitable parameters accordingly.
    /// </summary>
    public void checkLookatToggle(){
        PanelCamera        = GameObject.Find("/ConfigMenu/Panel/PanelCamera/Button/Panel");
        GameObject AgentID = GameObject.Find("/ConfigMenu/Panel/PanelCamera/Button/Panel/AgentID");
        
        Dropdown camSelect = PanelCamera.transform.Find("cameraType").GetComponent<Dropdown>();
        
        bool showLockFirstPerson = camSelect.options[camSelect.value].text == "First_Person" || camSelect.options[camSelect.value].text == "Torsum";
        bool activateLookAt =  camSelect.options[camSelect.value].text == "Look_At";
        bool activateFollow =  camSelect.options[camSelect.value].text == "Follow";

         if(camSelect.options[camSelect.value].text == "Look_At")
            PanelCamera.transform.Find("AgentID").GetComponent<InputField>().text = ConfigReader.camLookAtTarget.ToString();
        else if(camSelect.options[camSelect.value].text == "Follow" || camSelect.options[camSelect.value].text == "First_Person" || camSelect.options[camSelect.value].text == "Torsum")
            PanelCamera.transform.Find("AgentID").GetComponent<InputField>().text = ConfigReader.camFollowTarget.ToString();

        PanelCamera.transform.Find("AgentID").GetComponent<InputField>().interactable = activateLookAt||activateFollow||showLockFirstPerson;
        PanelCamera.transform.Find("FollowX").GetComponent<Toggle>().interactable = activateFollow;
        PanelCamera.transform.Find("FollowY").GetComponent<Toggle>().interactable = activateFollow;
        
        PanelCamera.transform.Find("LockPerson").GetComponent<Toggle>().interactable = showLockFirstPerson;
        PanelCamera.transform.Find("SmoothView").GetComponent<Toggle>().interactable = showLockFirstPerson;
    }
}