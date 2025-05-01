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
** Authors: Julien Bruneau
** Based on code from Unity-Technologies bitbucket (Image Synthesis for Machine Learning)
**
** Contact: crowd_group@inria.fr
*/

Shader "Filter/Filter_OpticalFlow"
{
	Properties
	{
		_Sensitivity("Sensitivity", Float) = 1
	}
		SubShader
	{
		// No culling or depth
		Cull Off ZWrite Off ZTest Always

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			#include "UnityCG.cginc"


			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
			};

			float4 _CameraMotionVectorsTexture_ST;
			v2f vert(appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = TRANSFORM_TEX(v.uv, _CameraMotionVectorsTexture);
				return o;
			}

			sampler2D _CameraMotionVectorsTexture;
			uniform float4 _ColorWheel[60];
			uniform uint _ncols;
			float _Sensitivity;
			uniform uint _MotionVector;


			float4 frag(v2f i) : SV_Target
			{
				float2 motion = tex2D(_CameraMotionVectorsTexture, i.uv).rg;
				
				motion.x = motion.x * _ScreenParams.x;
				motion.y = motion.y * _ScreenParams.y;

				if (_MotionVector>0)
					return float4(motion.x, motion.y, 0, 1);

				float rad = sqrt(motion.x*motion.x + motion.y*motion.y);
				float a = atan2(-motion.y, -motion.x) / UNITY_PI;

				float fk = (a + 1.0) / 2.0 * (_ncols - 1);
				uint k0 = (int)fk;
				uint k1 = (k0 + 1) % _ncols;
				float f = fk - k0;

				float4 c0 = _ColorWheel[k0];
				float4 c1 = _ColorWheel[k1];
				float4 col = (1 - f) * c0 + f * c1;

				//rad /= 10;
				//rad = pow(rad, 0.25);
				if (rad <= 1)
					col = 1 - rad * (1 - col); // increase saturation with radius
				else
					col *= .75; // out of range

				return float4(col.z, col.y, col.x, 1);
				
			}
			ENDCG
		}
	}
}
