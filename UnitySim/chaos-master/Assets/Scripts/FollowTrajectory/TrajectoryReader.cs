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
** Authors: Tristan Le Bouffant, Julian Joseph
**
** Contact: crowd_group@inria.fr
*/

using UnityEngine;
using System.Collections.Generic;
using System.Text;
using System.IO; 

public class TrajectoryReader
{
	[HideInInspector]public List<Vector4> mSkeleton;
	[HideInInspector]public List<Vector2> IndexList;

	[HideInInspector]public int mNbLines;
	public bool SimplifyCsv = false;
	public int nbSimplif = 2;
	int nbBoucle = 0;
	bool _Continue = true;
	public string _CsvFilename;
	
	public TrajectoryReader()
	{
	}
	public void Init(string CsvFilename, bool simplify_input, int nb_simplif, bool _MergeClosedPoints, float _ClosedPointsMinimumDistance)
	{
		if(CsvFilename == "")
		{
			return;
		}
		_CsvFilename = CsvFilename;
		SimplifyCsv = simplify_input;
		nbSimplif = nb_simplif;

		string line;
		mSkeleton = new List<Vector4>();
		IndexList = new List<Vector2>();
		// Create a new StreamReader, tell it which file to read and what encoding the file
		// was saved as
		StreamReader theReader = new StreamReader(_CsvFilename, Encoding.Default);
			
		// Immediately clean up the reader after this block of code is done.
		// You generally use the "using" statement for potentially memory-intensive objects
		// instead of relying on garbage collection.
		// (Do not confuse this with the using directive for namespace at the 
		// beginning of a class!)
		using (theReader)
		{
			// While there's lines left in the text file, do this:
			int iCurrentIndex = 0;
			//line = theReader.ReadLine(); // skip the first line
			do
			{
				line = theReader.ReadLine();
				if (line != null)
				{
					if(SimplifyCsv)
					{
						++nbBoucle;
						_Continue = (nbBoucle != nbSimplif);
						if(SimplifyCsv && _Continue)
						{
							continue;
						}
						nbBoucle = 0;
					}

					// Do whatever you need to do with the text line, it's a string now
					// In this example, I split it into arguments based on comma
					// deliniators, then send that array to DoStuff()
					//string[] entries = line.Split(';');
					//string[] entries = line.Split(',');
                    string[] entries = line.Split(new char[] { ',', ';' });

                    if (entries.Length > 0)
					{
                        if (!entries[1].Contains("#IND"))
                        {
                            Vector4 NewPoint = new Vector4(
                                float.Parse(entries[0], System.Globalization.CultureInfo.InvariantCulture),
                                float.Parse(entries[1], System.Globalization.CultureInfo.InvariantCulture),
                                float.Parse(entries[2], System.Globalization.CultureInfo.InvariantCulture),
                                (entries.Length > 3) ? float.Parse(entries[3], System.Globalization.CultureInfo.InvariantCulture) : float.NaN
                            );
                            if (_MergeClosedPoints && (mSkeleton.Count > 0))
                            {
                                Vector4 OldPoint = mSkeleton[mSkeleton.Count - 1];
                                if (float.IsNaN(OldPoint[3]))
                                    OldPoint[3] = 0;

                                Vector4 tmpNewPoint = new Vector4(NewPoint[0], NewPoint[1], NewPoint[2], NewPoint[3]);
                                tmpNewPoint[3] = (float.IsNaN(tmpNewPoint[3])) ? 0 : tmpNewPoint[3];

                                Vector4 CurVect = tmpNewPoint - OldPoint;
                                float distance = CurVect.magnitude;
                                if (distance > _ClosedPointsMinimumDistance)
                                {
                                    IndexList.Add(new Vector2(iCurrentIndex, float.Parse(entries[0], System.Globalization.CultureInfo.InvariantCulture)));
                                    mSkeleton.Add(NewPoint);
                                    ++iCurrentIndex;
                                }
                            }
                            else
                            {
                                IndexList.Add(new Vector2(iCurrentIndex, float.Parse(entries[0], System.Globalization.CultureInfo.InvariantCulture)));
                                mSkeleton.Add(NewPoint);
                                ++iCurrentIndex;
                            }
                        }
					}
				}
			}
			while (line != null);

			// Done reading, close the reader and return true to broadcast success    
			theReader.Close();
		}
	}
}
