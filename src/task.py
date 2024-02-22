"""
    This module contains dedicate methods for project tasks.
"""

import datetime
import os
import traceback

import src.resource.csv as csv
import src.resource.mp4 as mp4

def segmentation(csvPth: str, mp4Dir: str, outDir: str):
    """
        Segments the csv file by mp4 file names in datetime range format within the lookup directory,
        then output results into the output directory.

        Note: Assumes the first column being the filtering column.

        Keyword Arguments
        :csvPth (str) -- the source csv file path
        :mp4Dir (str) -- the lookup mp4 directory path
        :outDir (str) -- the output directory path
    """
    def parse_path_list(pthLst: list):
        """
            Parse mp4 file paths as timestamp range lists.

            Note: The file name is expected to have the following format,
                YYYYMMDDhhmmss_YYYYMMDDhhmmss, where
                YYYY is year, MM is month, DD is day,
                hh is hour, mm is minute and ss is second.

            Note: The file name is expected to have the following format,
                START_END, where
                START is the start time and END is the end time.

            Keyword Arguments
            :pthLst (list[str]) -- the list of mp4 file path
        """
        try:
            rngLst = []
            for pth in pthLst:
                pth = pth[0]
                try:
                    fil = os.path.basename(pth)
                    nam = fil.split('.')[0]

                    tokLst = nam.split('_')
                    for t in range(2):
                        tok = tokLst[t]
                        tokLst[t] = datetime.datetime(
                            int(tok[0:4]),
                            int(tok[4:6]),
                            int(tok[6:8]),
                            int(tok[8:10]),
                            int(tok[10:12]),
                            int(tok[12:14])
                        )
                        tokLst[t] = int(tokLst[t].timestamp())

                    rngLst.append(tokLst)
                except Exception:
                    continue
            
            return rngLst
        except Exception as exc:
            traceback.print_exception(exc)
            return None

    try:
        recLst = csv.read_as_list(csvPth)

        pthLst = mp4.list_path(mp4Dir)
        rngLst = parse_path_list(pthLst)

        for r in range(len(rngLst)):
            try:
                rng = rngLst[r]
                low = float(rng[0])
                upp = float(rng[1])

                segLst = []
                for rec in recLst:
                    try:
                        val = float(rec[0])
                        if (low <= val and val <= upp):
                            segLst.append(rec)
                    except Exception:
                        continue

                fil = os.path.basename(pthLst[r][0]).split('.')[0] + ".csv"
                pth = os.path.join(outDir, fil)
                csv.write_by_list(pth, segLst)
            except Exception as exc:
                continue
    except Exception as exc:
        traceback.print_exception(exc)
