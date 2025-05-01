using System.Collections;
using System.Collections.Generic;
using System.IO;
// using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using SFB;

public class InterfaceManager : MonoBehaviour
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
    LoadEnv env;                    // Contain datas for the environnement

    // Start is called before the first frame update
    void Start()
    {
        configPath = null;
        cam = Camera.main;

        menu  = GameObject.FindGameObjectWithTag("Menu");
        panel = GameObject.FindGameObjectWithTag("Panel");
        
        
        controlPanel = GameObject.Find("/controlCanvas/buttonControlPanel");
        controlPanel.SetActive(false);

        slider = GameObject.Find("/controlCanvas/Slider");
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

    // Update is called once per frame
    void Update()
    {
        if(controlPanel.activeSelf)
        {
            updateControlPanel();
        }
    }

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

    public void ReloadScenario(){
        //loadScenario(string trajDir);
        //tmpFollower.SetCsvFilename(f.FullName);
        //tmpFollower._SyncLaunchWithTrajectory = true; // start the character at the csv time
    }
}
