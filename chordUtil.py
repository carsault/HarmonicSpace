#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:38:02 2017

@author: tristan
"""

"""----------------------------------------------------------------------
-- Tristan Metadata and conv
----------------------------------------------------------------------"""

#%%
import json
import os
import numpy as np
import keras as K

def manual(tracks, userdir, audioSet, pitch = 6):
    for track in tracks.index:
        #fname = os.path.join( userdir + 'ismir2017_chords/working/chords/pump', os.path.extsep.join([track , 'npz']))
        fname = os.path.join( userdir + 'ismir2017_chords/working/chords/pump', os.path.extsep.join([track +"."+ str(pitch) , 'npz']))
        data = np.load(fname)
        d2 = dict(data)
        data.close()
        data = d2
        data['cqt/mag'] = data['cqt/mag'][0]
        audioSet.data.append(data['cqt/mag'])
        fname = json.load(open(userdir + 'ismir2017_chords/working/chords/augmentation/'+ track +"." + str(pitch) + ".jams"))
        #fname = json.load(open(userdir + 'ismir2017_chords/dataset/isophonics/metadata/Beatles/'+ track +".jams"))
        acc = {}
        acc['labels'] = []
        acc['timeStart'] = []
        acc['timeEnd'] = []
        u = fname
        for nbacc in range(len(u['annotations'][0]['data'])):
            acc['labels'].append(u['annotations'][0]['data'][nbacc]["value"])
            acc['timeStart'].append(u['annotations'][0]['data'][nbacc]["time"])
            acc['timeEnd'].append(u['annotations'][0]['data'][nbacc]["time"]+u['annotations'][0]['data'][nbacc]["duration"])
        audioSet.metadata['chord'].append(acc)
    return audioSet

def manualOne(track, userdir, audioSet, pitch = 6):
    #for track in tracks.index:
        #fname = os.path.join( userdir + 'ismir2017_chords/working/chords/pump', os.path.extsep.join([track , 'npz']))
    fname = os.path.join( userdir + 'ismir2017_chords/working/chords/pump', os.path.extsep.join([track +"."+ str(pitch) , 'npz']))
    data = np.load(fname)
    d2 = dict(data)
    data.close()
    data = d2
    data['cqt/mag'] = data['cqt/mag'][0]
    audioSet.data.append(data['cqt/mag'])
    fname = json.load(open(userdir + 'ismir2017_chords/working/chords/augmentation/'+ track +"." + str(pitch) + ".jams"))
    #fname = json.load(open(userdir + 'ismir2017_chords/dataset/isophonics/metadata/Beatles/'+ track +".jams"))
    acc = {}
    acc['labels'] = []
    acc['timeStart'] = []
    acc['timeEnd'] = []
    u = fname
    for nbacc in range(len(u['annotations'][0]['data'])):
        acc['labels'].append(u['annotations'][0]['data'][nbacc]["value"])
        acc['timeStart'].append(u['annotations'][0]['data'][nbacc]["time"])
        acc['timeEnd'].append(u['annotations'][0]['data'][nbacc]["time"]+u['annotations'][0]['data'][nbacc]["duration"])
    audioSet.metadata['chord'].append(acc)
    return audioSet

def convMetaLBC(audioSet,transformOptions):
    #convInput = []
    listBeatChord = [];
    hopSizeS = (transformOptions["hopSize"] / transformOptions["resampleTo"]);
    nbData = 0;
    curData = 0;
    audioSet.metadata['listBeatChord'] = {};
#Count the number of frames
    for k in range(len(audioSet.data)):
        nbData = nbData + len(audioSet.data[k]) - transformOptions["contextWindows"] + 1
#Pre-allocate the windowed dataset
        #local finalData = options.modelType == 'ladder' and torch.Tensor(nbData, nbBands * options.contextWindows) or torch.Tensor(nbData, nbBands, options.contextWindows);
        #finalData = np.array(nbData, nbBands, options.contextWindows)
    finalData = {}
    #finalLabels = {};
#-- Parse the whole set of windows
    for k in range(len(audioSet.data)):
        #print(k)
        maxFrame = len(audioSet.data[k])
        for numFrame in range(maxFrame - transformOptions["contextWindows"]  + 1):
            nbrAcc = 0
            while numFrame + (transformOptions["contextWindows"]  / 2) + 0.5 > (audioSet.metadata['chord'][k]['timeEnd'][nbrAcc] / hopSizeS) and nbrAcc+1 < len(audioSet.metadata['chord'][k]['timeStart']):
                nbrAcc = nbrAcc+1;            
            #finalLabels[curData] = list(audioSet.metadata['chord'][k][0]['labels'][nbrAcc]);
            finalData[curData] = audioSet.data[k][range(numFrame, numFrame + transformOptions["contextWindows"])];
            #finalData[curData] = (finalData[curData] - finalData[curData].mean()) / finalData[curData].max();
            listBeatChord.append(audioSet.metadata['chord'][k]['labels'][nbrAcc]);
            curData = curData + 1;
#audioSet.data[k] = convInputTens;
        audioSet.metadata['listBeatChord'][k] = listBeatChord;
        listBeatChord = []
    audioSet.data = finalData;
    #audioSet.metadata['listBeatChord'] = finalLabels
    return audioSet
#%%
QUALITIES = {
    #           1     2     3     4  5     6     7
    'maj':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'aug':     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'dim':     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4':    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus2':    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'maj6':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    'dim7':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'hdim7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    'maj9':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min9':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '9':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'b9':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '#9':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'min11':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '11':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '#11':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj13':   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min13':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '13':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'b13':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '1':       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '5':       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}


a0 = {
    'maj':     'maj',
    'min':     'min',
    'aug':     'N',
    'dim':     'N',
    'sus4':    'N',
    'sus2':    'N',
    '7':       'maj',
    'maj7':    'maj',
    'min7':    'min',
    'minmaj7': 'min',
    'maj6':    'maj',
    'min6':    'min',
    'dim7':    'N',
    'hdim7':   'N',
    'hdim':    'N',
    'maj9':    'maj',
    'min9':    'min',
    '9':       'maj',
    'b9':      'maj',
    '#9':      'maj',
    'min11':   'min',
    '11':      'maj',
    '#11':     'maj',
    'maj13':   'maj',
    'min13':   'min',
    '13':      'maj',
    'b13':     'maj',
    '1':       'N',
    '5':       'N',
    '': 'N'}

a1 = {
    'maj':     'maj',
    'min':     'min',
    'aug':     'N',
    'dim':     'dim',
    'sus4':    'N',
    'sus2':    'N',
    '7':       'maj',
    'maj7':    'maj',
    'min7':    'min',
    'minmaj7': 'min',
    'maj6':    'maj',
    'min6':    'min',
    'dim7':    'dim',
    'hdim7':   'dim',
    'hdim':    'dim',
    'maj9':    'maj',
    'min9':    'min',
    '9':       'maj',
    'b9':      'maj',
    '#9':      'maj',
    'min11':   'min',
    '11':      'maj',
    '#11':     'maj',
    'maj13':   'maj',
    'min13':   'min',
    '13':      'maj',
    'b13':     'maj',
    '1':       'N',
    '5':       'N',
    '': 'N'}

a2 = {
    'maj':     'maj',
    'min':     'min',
    'aug':     'N',
    'dim':     'dim',
    'sus4':    'N',
    'sus2':    'N',
    '7':       '7',
    'maj7':    'maj7',
    'min7':    'min7',
    'minmaj7': 'min',
    'maj6':    'maj',
    'min6':    'min',
    'dim7':    'dim7',
    'hdim7':   'dim',
    'hdim':    'dim',
    'maj9':    'maj7',
    'min9':    'min7',
    '9':       '7',
    'b9':      'maj',
    '#9':      'maj',
    'min11':   'min',
    '11':      'maj',
    '#11':     'maj',
    'maj13':   'maj',
    'min13':   'min',
    '13':      'maj',
    'b13':     'maj',
    '1':       'N',
    '5':       'N',
    '': 'N'}

a3 = {
    'maj':     'maj',
    'min':     'min',
    'aug':     'aug',
    'dim':     'dim',
    'sus4':    'sus',
    'sus2':    'sus',
    '7':       '7',
    'maj7':    'maj7',
    'min7':    'min7',
    'minmaj7': 'min',
    'maj6':    'maj',
    'min6':    'min',
    'dim7':    'dim7',
    'hdim7':   'dim',
    'hdim':    'dim',
    'maj9':    'maj7',
    'min9':    'min7',
    '9':       '7',
    'b9':      'maj',
    '#9':      'maj',
    'min11':   'min',
    '11':      'maj',
    '#11':     'maj',
    'maj13':   'maj',
    'min13':   'min',
    '13':      'maj',
    'b13':     'maj',
    '1':       'N',
    '5':       'N',
    '': 'N'}

gamme = {
    'Ab':   'G#',
    'A':    'A',
    'A#':   'A#',
    'Bb':   'A#',
    'B':    'B',
    'Cb':   'B',
    'C':    'C',
    'C#':   'C#',
    'Db':   'C#',
    'D':    'D',
    'D#':   'D#',
    'Eb':   'D#',
    'E':    'E',
    'F':    'F',
    'F#':   'F#',
    'Gb':   'F#',
    'G':    'G',
    'G#':   'G#',
    'N' :   'N',
    '' :    'N'}

tr = {
    'G':    'G#',
    'G#':   'A',
    'A':    'A#',
    'A#':   'B',
    'B':    'C',
    'C':    'C#',
    'C#':   'D',
    'D':    'D#',
    'D#':   'E',
    'E':    'F',
    'F':    'F#',
    'F#':   'G',
    'N' :   'N',
    '' :    'N'}

def getDictChord(alpha):
    chordList = []
    dictChord = {}
    for v in gamme.values():
        if v != 'N':
            for u in alpha.values():
                if u != 'N':
                    chordList.append(v+":"+u)
    chordList.append('N')
    #print(set(chordList))
    listChord = list(set(chordList))
    for i in range(len(listChord)):
        dictChord[listChord[i]] = i
    return dictChord, listChord

#dictA0 = getDictChord(a3)

def reduChord(initChord, alpha= 'a1', transp = 0):
    
    if initChord == "":
        print("buuug")
    initChord, bass = initChord.split("/") if "/" in initChord else (initChord, "")
    root, qual = initChord.split(":") if ":" in initChord else (initChord, "")
    root, noChord = root.split("(") if "(" in root else (root, "")
    qual, bass = qual.split("(") if "(" in qual else (qual, "")
      
    
    root = gamme[root]
    for i in range(transp):
        print("transpo")
        root = tr[root]
    
    if qual == "":
        if root == "N" or noChord != "":
            finalChord = "N"
        else:
            finalChord = root + ':maj'
    
    elif root == "N":
        finalChord = "N"
    
    else:
        if alpha == 'a1':
                qual = a1[qual]
        elif alpha == 'a0':
                qual = a0[qual]
        elif alpha == 'a2':
                qual = a2[qual]
        elif alpha == 'a3':
                qual = a3[qual]
        else:
                print("wrong alphabet value")
        if qual == "N":
            finalChord = "N"
        else:
            finalChord = root + ':' + qual

    return finalChord