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
** Authors: Fabrice ATREVI
**
** Contact: crowd_group@inria.fr
*/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

/// <summary>
/// Manage the welcome page: Show the logo of Chaos before to show the main menu
/// </summary>
public class LoadWelcome : MonoBehaviour
{
    public float timeToWait = 0.08f;              // Control the duration to show the logo

    private Image _backgroundImage;               // Background image
    private Text _text1, _text2;                  // Text to show the copyrigth and the definition of chaos
    private float _fading;                        // Control the fading speed
    private GameObject _currentPanel;             // Contain the current panel

    void Awake(){
        DontDestroyOnLoad(gameObject);
    }
    void Start()
    {
        StartCoroutine("showWelcome");
    }
/// <summary>
/// Coroutine to show the welcome panel
/// </summary>
    private IEnumerator showWelcome(){
        _currentPanel = GameObject.Find("Home/Panel"); // Avoid to use lot of tags.
        _backgroundImage = _currentPanel.transform.Find("Image").GetComponent<Image>();
        _text1 = _currentPanel.transform.Find("Text1").GetComponent<Text>();
        _text2 = _currentPanel.transform.Find("Text2").GetComponent<Text>();

        _backgroundImage.color = new Color(_backgroundImage.color.r,_backgroundImage.color.g,_backgroundImage.color.b,0f);
        _text1.color = new Color(1f,1f,1f,0f);
        _text2.color = new Color(1f,1f,1f,0f);

        while(_fading < 1)
        {
            yield return new WaitForSeconds(timeToWait);
            _fading  += 0.013f;
            _backgroundImage.color = new Color(_backgroundImage.color.r,_backgroundImage.color.g,_backgroundImage.color.b,_fading);
            _text1.color = new Color(_text1.color.r,_text1.color.g,_text1.color.b,_fading);
            _text2.color = new Color(_text2.color.r,_text2.color.g,_text2.color.b,_fading);
        }
        StartCoroutine("changeScene");
    }

    private IEnumerator changeScene(){
        while(_fading > 0)
        {
            yield return new WaitForSeconds(timeToWait);
            _fading  -= 0.019f;
            _backgroundImage.color = new Color(_backgroundImage.color.r,_backgroundImage.color.g,_backgroundImage.color.b,_fading);
            _text1.color = new Color(_text1.color.r,_text1.color.g,_text1.color.b,_fading);
            _text2.color = new Color(_text2.color.r,_text2.color.g,_text2.color.b,_fading);
        }
        SceneManager.LoadScene("startSc"); // Load the main menu
    }
}