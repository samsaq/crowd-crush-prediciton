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
** Authors: Tristan Le Bouffant, Julian Joseph, Fabrice ATREVI
**
** Contact: crowd_group@inria.fr
*/

using UnityEngine;
using Constants;

public class FollowTrajectory : MonoBehaviour 
{
	LineRenderer _LineRenderer = null;
	UnityEngine.AI.NavMeshAgent _ZeNavMeshAgent;
	TrajectoryReader _Myreader = null;
	float CurrentTime;
	
	[HideInInspector] public string _TrajectroryFilename;
	[HideInInspector] public bool _SyncLaunchWithTrajectory = true;

	[HideInInspector] public bool _SamplingInput = false;
	[HideInInspector] public int _SamplingFactor = 2;
	[HideInInspector] public GameObject _TargetToWatchForTheCamera;
	[HideInInspector] public float _DefaultAngularSpeed = 80.0f;
	[HideInInspector] public bool _DisplayLineRenderer;
	[HideInInspector] public bool _AnimateYAxe;
	// only if it's a player
	[HideInInspector] public float _AvatarDeltaYCorrection = 0.4f; // error when placing the gameObject. usefull for the locomotionSystem
	
	[HideInInspector] public bool _MergeClosedPoints = false;
	[HideInInspector] public float _ClosedPointsMinimumDistance;
	[HideInInspector] public int _NbPointsOnTheTrajectory;
		
	Vector4 vInf;
	Vector4 vSup;
	Vector3 vInfInWorld;
	Vector3 vSupInWorld;
	int iCurrentInf = 0;
	int iCurrentSup = 0;
	int _iCurrentIndex = 0;
	float fCurrentTime;
	float _fAngularSpeed;
	float _fOldAlpha; // beta tfloat _CurrentAlpha
	float _CurrentAlpha;
	float _Speed;
	bool _IsLaunchable = false;
	bool _bApplyFilter;
	bool _ChangingParent;
	Vector3 _InitialPositionWhenChangingParent;
	Vector3 _SavedParentPosition;
	bool _bIsDone;

	public float runTime = 0.0f;
	// optimization value
	int _ReaderIndexCount = 0;

	ObjectInterface _ObjectInterface = null;

	[HideInInspector] public AFilter _BasicFilter = new BasicFilter();
	[HideInInspector] public float _DeltaTimeForSmoothing = 0.5f;
	[HideInInspector] public FilteringMethod _FilterType = FilteringMethod.None;

	public void RegisterInterface(ObjectInterface new_interface)
	{
		_ObjectInterface = new_interface;
	}
	// Use this for initialization
	void Start () 
	{
        _bIsDone = false;
        _IsLaunchable = (_TrajectroryFilename != "" && _TrajectroryFilename != null);
        CurrentTime = 0.0f;

        iCurrentInf = 0;
        iCurrentSup = 1;
    }

    public void initNewTrajectories()
    {
        _bIsDone = false;
        _IsLaunchable = (_TrajectroryFilename != "" && _TrajectroryFilename != null);
        if (!_IsLaunchable)
        {
            Debug.LogError("Your CSV file is empty : " + _TrajectroryFilename);
			// show error message on the interface

            return;
        }

        CurrentTime = 0.0f;
        if (_Myreader == null)
        {
            _Myreader = new TrajectoryReader();
        }
        _Myreader.Init(_TrajectroryFilename, _SamplingInput, _SamplingFactor, _MergeClosedPoints, _ClosedPointsMinimumDistance);
        _ReaderIndexCount = _Myreader.IndexList.Count;
        InitLineRenderer();
        InitFilter();

        iCurrentInf = 0;
        iCurrentSup = 1;

        Vector4 iPosition = _Myreader.mSkeleton[0];
        fCurrentTime = iPosition[0];
        float yVal = 0.0f;
        if (float.IsNaN(iPosition[3]))
        {
            yVal = this.transform.localPosition.y;
            _AnimateYAxe = false;
            Debug.Log("A non alpha numerique value has been detected in your trajectory file " + _TrajectroryFilename + ". ensure x,y and z coordinates are set in the file if you want to animate Y coordinate. by default, the animation on Y axe is disable.");
        }
        else
        {
            yVal = (_AnimateYAxe) ? iPosition[3] : this.transform.localPosition.y;
        }
        this.transform.localPosition = new Vector3(iPosition[1], yVal, iPosition[2]);
        _ZeNavMeshAgent = this.GetComponent<UnityEngine.AI.NavMeshAgent>();
        if (_ZeNavMeshAgent)
        {
            _fAngularSpeed = _ZeNavMeshAgent.angularSpeed;
        }
        else
        {
            _fAngularSpeed = _DefaultAngularSpeed;
        }

        // init de la rotation
        Vector3 vPosCurrent = new Vector3(0.0f, 0.0f, 1.0f);
        Vector3 vPosSup = new Vector3(_Myreader.mSkeleton[1][1], 0.0f, _Myreader.mSkeleton[1][2]) - this.transform.localPosition; // y

        if (Vector3.Magnitude(vPosSup) != 0)
        {
            _fOldAlpha = 57.2957795f * Mathf.Acos(Vector3.Dot(vPosCurrent, vPosSup) / (Vector3.Magnitude(vPosCurrent) * Vector3.Magnitude(vPosSup)));
            if (Vector3.Cross(vPosCurrent, vPosSup).y < 0)
                _fOldAlpha *= -1;

            this.transform.rotation = Quaternion.Euler(0, _fOldAlpha/*+this.transform.rotation.eulerAngles.y*/, 0);
        }
        else
            _fOldAlpha = this.transform.rotation.eulerAngles.y;

        _SavedParentPosition = Vector3.zero;
        _InitialPositionWhenChangingParent = Vector3.zero;

        // init interface
        ObjectInterface MyObjInterface = new ObjectInterface();
        MyObjInterface.Init(this.gameObject);
        RegisterInterface(MyObjInterface);
    }

	public bool IsDone()
	{
		return _bIsDone;
	}

	public void InitFilter()
	{
		switch(_FilterType)
		{
		case FilteringMethod.None:
			_BasicFilter.Reader = _Myreader;
			_BasicFilter.IndexCount = _ReaderIndexCount;
			break;

		case FilteringMethod.Basic:
			_BasicFilter.Reader = _Myreader;
			_BasicFilter.IndexCount = _ReaderIndexCount;
			((BasicFilter)_BasicFilter)._DeltaTimeForSmoothing = _DeltaTimeForSmoothing;
			break;
			
		case FilteringMethod.LinearRegression:
		
			break;
			
		default:
			break;
		}
	}

	bool FindIntervalFromCurrentTime()
	{
		
		if(_iCurrentIndex>=_ReaderIndexCount)
		{
			return false;
		}
		for(int iIndex = _iCurrentIndex ; iIndex<_ReaderIndexCount ; ++iIndex)
		{
			float fCurrentIndexTime = 0.0f;
			float fNextIndexTime = 0.0f;
			
			if(iIndex+1 < _ReaderIndexCount)
			{
				fCurrentIndexTime = _Myreader.IndexList[iIndex][1];
				fNextIndexTime    = _Myreader.IndexList[iIndex+1][1];
				
				if( (fCurrentTime>=fCurrentIndexTime) && (fCurrentTime<fNextIndexTime))
				{
					_iCurrentIndex = iIndex;
					iCurrentInf = _iCurrentIndex;
					iCurrentSup = _iCurrentIndex+1;
					
					vInf = _Myreader.mSkeleton[iCurrentInf];
					vSup = _Myreader.mSkeleton[iCurrentSup];
					vInfInWorld.x = vInf[1];
					if(_AnimateYAxe)
						vInfInWorld.y =  vInf[3];
					vInfInWorld.z = vInf[2];
					
					vSupInWorld.x = vSup[1];
					if(_AnimateYAxe)
						vSupInWorld.y = vSup[3];
					vSupInWorld.z = vSup[2];

					return true;
				}
			}
		}
		return false;
	}
	
	public void  SetApplyFilter(bool apply_filter)
	{
		_bApplyFilter = apply_filter;
	}
	
	AFilter GetFilter()
	{
		AFilter filter = null;

		switch(_FilterType)
		{
		case FilteringMethod.None:
			filter = null;
			break;

		case FilteringMethod.Basic:
			filter = _BasicFilter;
			break;

		case FilteringMethod.LinearRegression:
			filter = _BasicFilter;
			break;

		default:
			filter = _BasicFilter;
			break;
		}

		return filter;
	}

	void UpdateOrientation(Vector3 LocalTargetInfOnTrajectory, Vector3 LocalTargetSupOnTrajectory)
	{

		if(!_bApplyFilter)
		{
			LocalTargetInfOnTrajectory = this.transform.localPosition;
			LocalTargetSupOnTrajectory = vSupInWorld;
		}

		// 1-calcul de l'angle entre la position actuelle et la destination
		Vector3 vPosCurrent = new Vector3(0.0f, 0.0f, 1.0f);//this.transform.localPosition; // x
		Vector3 vPosSup = new Vector3(LocalTargetSupOnTrajectory.x, 0.0f, LocalTargetSupOnTrajectory.z) -  new Vector3(LocalTargetInfOnTrajectory.x, 0.0f, LocalTargetInfOnTrajectory.z); // new Vector3(_Myreader.mSkeleton[iCurrentSup][1], 0.0f, _Myreader.mSkeleton[iCurrentSup][2]) -  LocalTargetInfOnTrajectory; // y

		// if stay inplace, don't update orientation
		if((LocalTargetSupOnTrajectory.x == LocalTargetInfOnTrajectory.x) && (LocalTargetSupOnTrajectory.z == LocalTargetInfOnTrajectory.z))
		{
			return;
		}
		// in radian
		_CurrentAlpha = devUnity.TrigoTools.RadianToDegree(Mathf.Acos(Vector3.Dot(vPosCurrent, vPosSup)/(Vector3.Magnitude(vPosCurrent)*Vector3.Magnitude(vPosSup))));
		if(Vector3.Cross(vPosCurrent, vPosSup).y<0)
			_CurrentAlpha *= -1;

		if(float.IsNaN(_CurrentAlpha))
		{
			Debug.LogWarning("LineRendererTraj::UpdateOrientation : currentAlpha is invalid");
			return;
		}
		float MaxAngularForDeltaTime = Time.deltaTime * _fAngularSpeed;
		
		_CurrentAlpha = devUnity.TrigoTools.RadianToDegree(devUnity.TrigoTools.getAngleInTrigoRad(devUnity.TrigoTools.DegreeToRadian(_CurrentAlpha)));
		if(Mathf.Abs(_CurrentAlpha-_fOldAlpha)>MaxAngularForDeltaTime)
		{
			if(_CurrentAlpha>_fOldAlpha)
			{
				if(_fOldAlpha+devUnity.TrigoTools.getPiInDegree()>=_CurrentAlpha)
					_CurrentAlpha = _fOldAlpha+MaxAngularForDeltaTime;
				else
					_CurrentAlpha = _fOldAlpha-MaxAngularForDeltaTime;
			}
			else
			{
				if(_fOldAlpha>=_CurrentAlpha+devUnity.TrigoTools.getPiInDegree())
					_CurrentAlpha = _fOldAlpha+MaxAngularForDeltaTime;
				else
					_CurrentAlpha = _fOldAlpha-MaxAngularForDeltaTime;
			}
		}
		_CurrentAlpha = devUnity.TrigoTools.RadianToDegree(devUnity.TrigoTools.getAngleInTrigoRad(devUnity.TrigoTools.DegreeToRadian(_CurrentAlpha)));
		_fOldAlpha = _CurrentAlpha;
		this.transform.rotation = Quaternion.Euler(0.0f, _CurrentAlpha, 0.0f);
	}
	
	/*float ComputeDeltaBeetween2Angles(float orig_orientation, float target_orientation)
	{
		// transformation en 0,2pi
		float d_orig_orientation2 = getAngleInTrigoRad(orig_orientation);
		float destination_alpha = getAngleInTrigoRad(target_orientation);
		
		float delta_alpha = destination_alpha - d_orig_orientation2;
		
		float delta_alpha2 = (delta_alpha>getPi()) ? delta_alpha-2*getPi() : delta_alpha;
		float delta_alpha3 = (delta_alpha2<-getPi()) ? 2*getPi()+delta_alpha2 : delta_alpha2;
		float delta_alpha4 = devUnity.TrigoTools.RadianToDegree(delta_alpha3);
		return delta_alpha4;
	}*/ // TODO: cleanup after refactor

	// Update is called once per frame
	void Update () 
	{
		if(!_IsLaunchable || _bIsDone)
			return;

		if(_SyncLaunchWithTrajectory && CurrentTime < _Myreader.mSkeleton[0][0] && 0.2 < _Myreader.mSkeleton[0][0])
		{
			CurrentTime += Time.deltaTime;
			_ObjectInterface.Hide();
			return;
		}
        else if (fCurrentTime > _Myreader.mSkeleton[_Myreader.mSkeleton.Count-1][0])
        {
            GameObject.Destroy(this.gameObject);
            return;
        }
		else
			_ObjectInterface.Show();
        
        if (!_ChangingParent && transform.parent != null)
		{
			if(_InitialPositionWhenChangingParent == Vector3.zero)
			{
				_InitialPositionWhenChangingParent = transform.parent.position;
			}
			_ChangingParent = true;
		}
		else if(_ChangingParent && transform.parent == null)
		{
			// to do
			//this.transform.localPosition = this.transform.localPosition+_SavedParentPosition;
			_ChangingParent = false;
			//_InitialPositionWhenChangingParent = Vector3.zero;
		}

		fCurrentTime += Time.deltaTime;
		runTime = (float)_iCurrentIndex /_ReaderIndexCount;

		_bIsDone = !FindIntervalFromCurrentTime();
		
		if(_bIsDone)
		{
			LoadEnv.cmptDone++;
			
			// Show the replay button on the scene
            if(LoadEnv.cmptDone == LoadEnv.nbAgent)
			{
                MenuManager.replayButton.SetActive(true);
				GameObject[] list = GameObject.FindGameObjectsWithTag("Player");
            	foreach(GameObject a in list)
                	Destroy(a);
			}
			return;
		}

		// update position
		//CharacterController ZeCharCon = GetComponent<CharacterController>();
		//ZeCharCon.center = new Vector3(ZeCharCon.center.x, ZeCharCon.center.y+ _AvatarDeltaYCorrection, ZeCharCon.center.z);

		
		Vector3 LocalTargetInfOnTrajectory = new Vector3();
		Vector3 LocalTargetSupOnTrajectory = new Vector3();
		Vector3 newPos;
        //bool hide = false;
        //if (vInfInWorld[0] < -14|| vInfInWorld[0] > 14|| vInfInWorld.y< -14 || vInfInWorld.y> 14)
        //    hide = true;
        //if (vSupInWorld[0] < -14|| vSupInWorld[0] > 14|| vSupInWorld.y< -14 || vSupInWorld.y> 14)
        //    hide = true;
        //Debug.Log(vInfInWorld + "---" + vSupInWorld);
        if (_bApplyFilter)
		{
			GetFilter().SmootPosition(fCurrentTime, _iCurrentIndex, out LocalTargetInfOnTrajectory, out LocalTargetSupOnTrajectory);
			newPos = (LocalTargetInfOnTrajectory+LocalTargetSupOnTrajectory)/2.0f;
		}
		else
		{
			float TVInf = vInf[0];
			float TVSup = vSup[0];
			float TReference = TVSup-TVInf;
			float TInInterval = fCurrentTime-TVInf;
			float RatioTimeInInterval = TInInterval/TReference;
			Vector3 DirectionOfTheInterval = vSupInWorld - vInfInWorld;
			Vector3 CurrentPositionOnTheVector = RatioTimeInInterval*DirectionOfTheInterval; 
			newPos = vInfInWorld+CurrentPositionOnTheVector;
		}

		Vector3 oldPos = new Vector3(this.transform.localPosition.x, 0.0f, this.transform.localPosition.z);
		_Speed = Vector3.Magnitude(oldPos - newPos) / Time.deltaTime;
		//_Speed = Vector3.Magnitude(vSupInWorld - vInfInWorld) / Time.deltaTime;
        //Debug.Log("_Speed => " + _Speed);


        if (_Speed>5 && Time.deltaTime>0)
        {

            //Debug.Log("hide => " + hide);
            for (int i = 0; i < transform.childCount; ++i)
            {
                transform.GetChild(i).gameObject.SetActive(false);
            }
            foreach (Transform child in transform)
            {
                //Debug.Log(child.gameObject);
                Renderer rend = child.gameObject.GetComponent<Renderer>();
                if (rend != null)
                {
                    rend.enabled = false;
                }
            }
        }
        else
        {
            for (int i = 0; i < transform.childCount; ++i)
            {
                transform.GetChild(i).gameObject.SetActive(true);
            }
            foreach (Transform child in transform)
            {
                //Debug.Log(child.gameObject);
                Renderer rend = child.gameObject.GetComponent<Renderer>();
                if (rend != null)
                {
                    rend.enabled = true;
                }
            }
        }

        if (_Speed > 0)
		{
			// update position
			//float yVal = (_AnimateYAxe) ? newPos.y : this.transform.localPosition.y+_AvatarDeltaYCorrection;
			float yVal = (_AnimateYAxe) ? newPos.y : this.transform.localPosition.y;
            //yVal = (hide) ? -1000 : 0;
            if (transform.parent)
			{
				_SavedParentPosition = transform.parent.position-_InitialPositionWhenChangingParent;
				this.transform.localPosition = new Vector3(newPos.x, yVal, newPos.z)-_InitialPositionWhenChangingParent;
			}
			else
				this.transform.localPosition = new Vector3(newPos.x+_SavedParentPosition.x, yVal, newPos.z+_SavedParentPosition.z);

			if(_TargetToWatchForTheCamera == null)
			{
				// update orientation
				UpdateOrientation(LocalTargetInfOnTrajectory, LocalTargetSupOnTrajectory);
			}
			else
			{
				this.transform.LookAt(_TargetToWatchForTheCamera.transform.position);
			}
		}
	}
	
	public void InitLineRenderer()
	{
		if(_LineRenderer == null)
		{
			//_LineRenderer = new LineRenderer();
			this.gameObject.AddComponent<LineRenderer>();
			_LineRenderer = GetComponent<LineRenderer>();
			_LineRenderer.hideFlags = HideFlags.HideInInspector;
		}
		SetDisplayLineRendererProperty(_DisplayLineRenderer);
		_LineRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
		_LineRenderer.receiveShadows = false;
		_LineRenderer.SetWidth(0.1f, 0.1f);
		int iVerticesCount = _Myreader.mSkeleton.Count;
		//_Config._LineRenderer = this._IrisSurfacePrefab.GetComponent("LineRenderer") as LineRenderer;
		_LineRenderer.SetVertexCount(iVerticesCount); //+1 pour boucler le premier vertex avec le dernier
		for(int j=0 ; j<iVerticesCount ; j++)
		{
			Vector3 iPosition = _Myreader.mSkeleton[j];
			
			// construction du line renderer
			_LineRenderer.SetPosition(j, new Vector3(iPosition[1], 0.0f, iPosition[2]));
		}
		_NbPointsOnTheTrajectory = iVerticesCount;
	}
	public void ForceDisplayLineRendererProperty(bool displayLineRenderer)
	{
		_DisplayLineRenderer = displayLineRenderer;
		SetDisplayLineRendererProperty(_DisplayLineRenderer);
	}
	public void SetDisplayLineRendererProperty(bool displayLineRenderer)
	{
		if(_LineRenderer)
			_LineRenderer.enabled = displayLineRenderer;
	}
	public void ChangeAngularSpeed(float angular_speed)
	{
		_fAngularSpeed = angular_speed;
	}
	public void SetCsvFilename(string csv_filename)
	{
		_TrajectroryFilename = csv_filename;
        initNewTrajectories();
	}
}
