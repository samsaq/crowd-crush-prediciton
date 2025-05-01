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
** Authors: Julien Bruneau, Florian Berton, Fabrice Atrevi
** Using of the standalonebrower package of Gökhan Gökçe: Copyright (c) 2017 Gökhan Gökçe
**
** Contact: crowd_group@inria.fr
*/

using System.Collections;
using System.Collections.Generic;
using System.IO;
// using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using SFB;

/// <summary>
/// Starting menu manager
/// </summary>
public class MenuManager : MonoBehaviour
{
    Text pathText;                  // used to show the path of the selected scenario folder
    string scenarioFolderPath;      // path to the selected scenario folder
    string configPath;              // path to the selected scenario file
    ObstaclesReader obstReader;     // object loading/creating obstacles

    Camera cam;                     // the main camera
    Dropdown configFilesMenu;       // GUI object to select the scenario
    GameObject menu;                // the menu gameObject
    public GameObject panel;                  // the panel gameObject
    public GameObject cameraMovPanel;         // Panel of camera mouvement
    public static GameObject controlPanel;    // The play, pause and stop control panel
    public GameObject configPanel;  // the panel to set the config parameters
    public GameObject topMenuPanel; // the top menu panel
    public GameObject slider;       // Slider for the progress state bar
    public Slider progressBar;

    public static GameObject replayButton;
    public LoadEnv env;                    // Contain datas for the environnement

    /// <summary>
    /// initialize the menu
    /// </summary>
	void Start()
    {
        configPath = null;
        cam = Camera.main;

        menu  = GameObject.FindGameObjectWithTag("Menu");
        panel = GameObject.FindGameObjectWithTag("Panel");
        
		replayButton = GameObject.Find("/controlCanvas/Replay");
        replayButton.SetActive(false);
        
        controlPanel = GameObject.Find("/controlCanvas/buttonControlPanel");
        controlPanel.SetActive(false);

        slider = GameObject.Find("/controlCanvas/progressBar");
        progressBar = GameObject.Find("/controlCanvas").transform.Find("progressBar").GetComponent<Slider>();
        slider.SetActive(false);

        topMenuPanel = GameObject.Find("/controlCanvas/fakePanel");
        topMenuPanel.SetActive(false);

        cameraMovPanel = GameObject.Find("/controlCanvas/cameraMovPanel");
        
        cameraMovPanel.SetActive(false);

        configPanel = GameObject.FindGameObjectWithTag("configPanel");

        obstReader = new ObstaclesReader();

        configFilesMenu = panel.GetComponentInChildren<Dropdown>();
        if(configFilesMenu!=null)
            configFilesMenu.options.Clear();

        Time.timeScale = 0;
    }

    /// <summary>
    /// Update at each frame
    /// </summary>
    void Update()
    {
        if(controlPanel.activeSelf)
        {
            updateControlPanel();
        }
        if(cameraMovPanel.activeSelf)
        {
            checkLookatToggleOnScene();
        }
        if(slider.activeSelf){
            GameObject[] list = GameObject.FindGameObjectsWithTag("Player");
            float[] tab = new float[list.Length];
            float t = 0f;
            for(int i = 0; i < list.Length; i++)
            {
                t += list[i].GetComponent<FollowTrajectory>().runTime/list.Length;
            }
            progressBar.value = t;
        }
            
    }

    /// <summary>
    /// Update the highlight active button on the control panel
    /// </summary>
    public void updateControlPanel()
    {
        ColorBlock selectedColor, disableColor;
        
        selectedColor = controlPanel.transform.Find("play").GetComponent<Button>().colors;
        disableColor = controlPanel.transform.Find("play").GetComponent<Button>().colors;
        
        selectedColor.normalColor = new Color32(212,61,61,255);
        disableColor.normalColor = new Color32(200,200,200,255);
        
        if(Time.timeScale == 1)
        {
            controlPanel.transform.Find("play").GetComponent<Button>().colors = selectedColor;
            controlPanel.transform.Find("pause").GetComponent<Button>().colors = disableColor;
        }
        else{
            controlPanel.transform.Find("play").GetComponent<Button>().colors = disableColor;
            controlPanel.transform.Find("pause").GetComponent<Button>().colors = selectedColor;
        }
    }

    /// <summary>
    /// Function to play the scenario
    /// </summary>
    public void Play()
    {
        Time.timeScale = 1;
    }
    
    /// <summary>
    /// Function to pause the scenario
    /// </summary>
    public void Pause()
    {
        Time.timeScale = 0;
    }

    /// <summary>
    /// Open the folder explorer to select the scenario folder
    /// Based on the standalonebrower package of Gökhan Gökçe
    /// Copyright (c) 2017 Gökhan Gökçe
    /// </summary>
    /// <returns>selected scenario'S folder path</returns>
    public void OpenExplorer()
    {
        var folderPath = StandaloneFileBrowser.OpenFolderPanel("Choose the scenario's folder", "", false);
        if (folderPath.Length == 0) {
            return;
        }

        scenarioFolderPath = "";
        foreach (var p in folderPath) {
            scenarioFolderPath += p;
        }

        if (scenarioFolderPath != null)
        {
            pathText = panel.transform.Find("path").GetComponent<Text>();
            pathText.text = scenarioFolderPath;

            loadConfigFileList();
        }
    }

    /// <summary>
    /// Check if argument are given and start scenario if needed
    /// command line: ChAOS.exe -s ScenarioPath -r width(optional) height(optional)
    /// </summary>
    public void CheckCommandLine()
    {
        string [] arguments = System.Environment.GetCommandLineArgs();
        if (arguments.Length >1 && !Application.isEditor)
        {
            // Parse Command Line
            CommandLineParser CommandLine = new CommandLineParser(arguments);
            configPath = CommandLine.GetScenarioFile();
            
            // StartScenario
            ConfigReader.LoadConfig(configPath);
            
            obstReader.clear();
            obstReader.createObstacles(ConfigReader.obstaclesFile, ConfigReader.stageInfos);

            env = gameObject.GetComponent<LoadEnv>();
            env.loadScenario(ConfigReader.trajectoriesDir);
            
            //menu.SetActive(false);
            //configPanel.SetActive(false);
            //controlPanel.SetActive(false);
            //slider.SetActive(false);
            //topMenuPanel.SetActive(false);
            //cameraMovPanel.SetActive(false);
            
            Time.timeScale = 1;
        }
    }

    /// <summary>
    /// update the dropdown menu listing the scenarios
    /// </summary>
    public void loadConfigFileList()
    {
        if (configFilesMenu == null)
            return;

        DirectoryInfo dir;
        dir = new DirectoryInfo(scenarioFolderPath);
        
        FileInfo[] infos = dir.GetFiles("*.xml");
        configFilesMenu.options.Clear();

        foreach (FileInfo i in infos)
        {
            configFilesMenu.options.Add(new Dropdown.OptionData(i.Name.Remove(i.Name.Length - 4)));
        }
        configFilesMenu.value = 0;
        configFilesMenu.RefreshShownValue();

        updateConfigFile();
    }

    /// <summary>
    /// Update the current scenario file
    /// </summary>
    public void updateConfigFile()
    {
        configPath = scenarioFolderPath + '/' + configFilesMenu.options[configFilesMenu.value].text + ".xml";
    }

    public void showCameraMovPanel(){
        cameraMovPanel.SetActive(true);
        cameraMovPanel.transform.Find("cameraType").GetComponent<Dropdown>().value = ConfigReader.camType;
        
        Dropdown camSelect = cameraMovPanel.transform.Find("cameraType").GetComponent<Dropdown>();
        
        if(camSelect.options[camSelect.value].text == "Look_At")
            cameraMovPanel.transform.Find("AgentID").GetComponent<InputField>().text = ConfigReader.camLookAtTarget.ToString();
        else if(camSelect.options[camSelect.value].text == "Follow" || camSelect.options[camSelect.value].text == "First_Person" || camSelect.options[camSelect.value].text == "Torsum")
            cameraMovPanel.transform.Find("AgentID").GetComponent<InputField>().text = ConfigReader.camFollowTarget.ToString();
        
        cameraMovPanel.transform.Find("FollowX").GetComponent<Toggle>().isOn    = ConfigReader.camFollowOnX;
        cameraMovPanel.transform.Find("FollowY").GetComponent<Toggle>().isOn    = ConfigReader.camFollowOnY;
        cameraMovPanel.transform.Find("LockPerson").GetComponent<Toggle>().isOn = ConfigReader.camLockFirstPerson;
        cameraMovPanel.transform.Find("SmoothView").GetComponent<Toggle>().isOn = ConfigReader.camSmoothFirstPerson;
    }
    /// <summary>
    /// Load the current scenario file and remove the menu
    /// </summary>
    public void startScenario()
    {
        if (configPath == null)
            return;
    
        ConfigReader.LoadConfig(configPath);
        obstReader.clear();
        
        obstReader.createObstacles(ConfigReader.obstaclesFile, ConfigReader.stageInfos);

        env = gameObject.GetComponent<LoadEnv>();
        env.loadScenario(ConfigReader.trajectoriesDir);
        
        panel.SetActive(false);
        
        configPanel.SetActive(false);
        
        controlPanel.SetActive(true);
        
        topMenuPanel.SetActive(true);
        
        // Show the camera movement parameters on the scene to allow user to play with during the animation
        showCameraMovPanel();
        checkLookatToggleOnScene();

        slider.SetActive(true);
        GameObject.Find("/controlCanvas/fakePanel/topMenu").SetActive(false);
    }
   
    /// <summary>
    /// Check the toggle button on the camera movement panel
    /// </summary>
    public void checkLookatToggleOnScene(){
        GameObject PanelMovCamera = GameObject.Find("/controlCanvas/cameraMovPanel");
        Dropdown camSelect = PanelMovCamera.transform.Find("cameraType").GetComponent<Dropdown>();
        
        bool showLockFirstPerson = camSelect.options[camSelect.value].text == "First_Person" || camSelect.options[camSelect.value].text == "Torsum";
        
        bool activateLookAt =  camSelect.options[camSelect.value].text == "Look_At";
        bool activateFollow =  camSelect.options[camSelect.value].text == "Follow";

        PanelMovCamera.transform.Find("AgentID").GetComponent<InputField>().interactable = activateLookAt||activateFollow||showLockFirstPerson;
        PanelMovCamera.transform.Find("FollowX").GetComponent<Toggle>().interactable     = activateFollow;
        PanelMovCamera.transform.Find("FollowY").GetComponent<Toggle>().interactable     = activateFollow;
        
        PanelMovCamera.transform.Find("LockPerson").GetComponent<Toggle>().interactable  = showLockFirstPerson;
        PanelMovCamera.transform.Find("SmoothView").GetComponent<Toggle>().interactable  = showLockFirstPerson;
    }
    
    /// <summary>
    /// show/hide menu
    /// </summary>
    public void toogleMenu()
    {
        //if(panel.activeSelf==false)
        //{
        panel.SetActive(!panel.activeSelf);
        controlPanel.SetActive(!panel.activeSelf);
        topMenuPanel.SetActive(!panel.activeSelf);
        cameraMovPanel.SetActive(!panel.activeSelf);
        slider.SetActive(!panel.activeSelf);
        replayButton.SetActive(!panel.activeSelf);

        Pause();

        //GameObject[] list = GameObject.FindGameObjectsWithTag("Player");
        //foreach(GameObject a in list)
        //    Destroy(a);

        //Time.timeScale = 0;
        //}

    }

    /// <summary>
    /// show the config parameters panel for setting
    /// </summary>
    public void configParams()
    {
        Time.timeScale = 0; // Pause the scenario when modifying the parameters
        configPanel.SetActive(!configPanel.activeSelf);
        controlPanel.SetActive(!controlPanel.activeSelf);
        slider.SetActive(!slider.activeSelf);
        topMenuPanel.SetActive(false);
        cameraMovPanel.SetActive(false);

        if(replayButton.activeSelf)
            replayButton.SetActive(false);

        ConfigManager.showConfig();
    }

    /// <summary>
    /// save the current scenario file
    /// </summary>
    public void saveConfig()
    {
        ConfigManager.SaveConfig(configPath);

        configPanel.SetActive(!configPanel.activeSelf);
        
        ConfigReader.LoadConfig(configPath);
        env.updateScenario(ConfigReader.trajectoriesDir);
        controlPanel.SetActive(!controlPanel.activeSelf);
        slider.SetActive(!panel.activeSelf);
        topMenuPanel.SetActive(true);
        cameraMovPanel.SetActive(true);

        if(LoadEnv.cmptDone == LoadEnv.nbAgent)
            replayButton.SetActive(true);

        // Update Camera type on the screen
        if( cameraMovPanel.activeSelf)
            showCameraMovPanel();
    }

    /// <summary>
    /// close the config panel without saving the parameters 
    /// </summary>
    public void cancelConfig()
    {
        configPanel.SetActive(!configPanel.activeSelf);
        controlPanel.SetActive(!controlPanel.activeSelf);

        slider.SetActive(!panel.activeSelf);
        topMenuPanel.SetActive(true);
        cameraMovPanel.SetActive(true);

        if(LoadEnv.cmptDone == LoadEnv.nbAgent)
            replayButton.SetActive(true);
    }

    public void closeHelpAboutPanel()
    {
        if(LoadEnv.cmptDone == LoadEnv.nbAgent)
            replayButton.SetActive(true);
    }
    /// <summary>
    /// quit the application
    /// </summary>
    public void exit()
    {
        Application.Quit();
    }

    /// <summary>
    /// Replay the scenario
    /// </summary>
    public void replayScenario()
    {
        startScenario();
        replayButton.SetActive(false);
        Pause();
    }
}