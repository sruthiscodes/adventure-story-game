import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import styled from '@emotion/styled';
import { useGameStore } from './store/gameStore';
import { initializeAI } from './utils/aiService';

// Components
import Welcome from './components/Welcome';
import CharacterCreation from './components/CharacterCreation';
import Gameplay from './components/Gameplay';

function App() {
  const { 
    gameStarted, 
    character, 
    initializeGame, 
    startNewGame, 
    loadGame,
    clearPersistedState
  } = useGameStore();
  
  const [appState, setAppState] = useState<'loading' | 'welcome' | 'character-creation' | 'gameplay'>('loading');
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [isMultiPageMode, setIsMultiPageMode] = useState(false);
  
  // Initialize AI models on app load
  useEffect(() => {
    const initialize = async () => {
      try {
        await initializeAI();
        
        // Check if we should use multi-page mode
        const gameMode = localStorage.getItem('game-mode') || 'single-page';
        setIsMultiPageMode(gameMode === 'multi-page');
        
        // Check for special flow fix from sessionStorage
        const fixFlow = sessionStorage.getItem('fix-game-flow');
        if (fixFlow) {
          sessionStorage.removeItem('fix-game-flow');
          setAppState(fixFlow as any);
          return;
        }
        
        // Force welcome screen on first load
        const urlParams = new URLSearchParams(window.location.search);
        const forceReset = urlParams.get('reset') === 'true';
        
        if (forceReset) {
          clearPersistedState();
          setAppState('welcome');
          return;
        }
        
        // Check if there's a valid saved game
        const hasSavedGame = loadGame();
        
        if (hasSavedGame && gameStarted && character.name) {
          setAppState('gameplay');
        } else {
          // Reset to initial state if no valid saved game or incomplete data
          clearPersistedState();
          setAppState('welcome');
        }
      } catch (error) {
        console.error('Failed to initialize AI:', error);
        setAppState('welcome');
      }
    };
    
    initialize();
  }, []);  // Only run on mount to prevent loops
  
  const handleNewGame = () => {
    setIsTransitioning(true);
    initializeGame();
    setAppState('character-creation');
    setIsTransitioning(false);
  };
  
  const handleLoadGame = () => {
    setIsTransitioning(true);
    if (loadGame()) {
      setAppState('gameplay');
    }
    setIsTransitioning(false);
  };
  
  const handleCreateCharacter = (newCharacter: any) => {
    setIsTransitioning(true);
    console.log("Creating character:", newCharacter);
    
    // Store in localStorage directly to ensure it persists
    try {
      startNewGame(newCharacter);
      
      // Use setTimeout to ensure state updates have time to process
      setTimeout(() => {
        setAppState('gameplay');
        setIsTransitioning(false);
      }, 100);
    } catch (error) {
      console.error("Error starting game:", error);
      setIsTransitioning(false);
      alert("There was an error creating your character. Please try again.");
    }
  };
  
  const renderContent = () => {
    // Don't change state during transitions
    if (isTransitioning) {
      return <LoadingScreen>Loading...</LoadingScreen>;
    }
    
    // For multi-page mode, we use separate pages with full page refresh
    if (isMultiPageMode) {
      // Save the current state to localStorage for page transitions
      localStorage.setItem('current-page', appState);
      
      // Render the appropriate page and provide navigation functions
      switch (appState) {
        case 'loading':
          return <LoadingScreen>Initializing AI models...</LoadingScreen>;
        case 'welcome':
          return (
            <Welcome 
              onNewGame={() => {
                initializeGame();
                localStorage.setItem('current-page', 'character-creation');
                window.location.reload();
              }} 
              onLoadGame={() => {
                if (loadGame()) {
                  localStorage.setItem('current-page', 'gameplay');
                  window.location.reload();
                }
              }} 
            />
          );
        case 'character-creation':
          return (
            <CharacterCreation 
              onCreateCharacter={(newCharacter) => {
                console.log("Creating character:", newCharacter);
                try {
                  startNewGame(newCharacter);
                  localStorage.setItem('current-page', 'gameplay');
                  window.location.reload();
                } catch (error) {
                  console.error("Error starting game:", error);
                  alert("There was an error creating your character. Please try again.");
                }
              }} 
            />
          );
        case 'gameplay':
          return <Gameplay />;
        default:
          localStorage.setItem('current-page', 'welcome');
          window.location.reload();
          return <LoadingScreen>Redirecting...</LoadingScreen>;
      }
    }
    
    // Original single-page application flow
    switch (appState) {
      case 'loading':
        return <LoadingScreen>Initializing AI models...</LoadingScreen>;
      case 'welcome':
        return <Welcome onNewGame={handleNewGame} onLoadGame={handleLoadGame} />;
      case 'character-creation':
        return <CharacterCreation onCreateCharacter={handleCreateCharacter} />;
      case 'gameplay':
        return <Gameplay />;
      default:
        return <Navigate to="/" />;
    }
  };
  
  return (
    <AppContainer>
      <BackgroundOverlay />
      <ContentWrapper>
        {renderContent()}
      </ContentWrapper>
    </AppContainer>
  );
}

const AppContainer = styled.div`
  position: relative;
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #121212;
  color: #f0f0f0;
`;

const ContentWrapper = styled.div`
  position: relative;
  z-index: 1;
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
`;

const BackgroundOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: radial-gradient(circle at center, rgba(50, 50, 50, 0.4) 0%, rgba(0, 0, 0, 0.8) 100%);
  z-index: 0;
`;

const LoadingScreen = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  font-size: 1.5rem;
  color: #ffd700;
  position: relative;
  z-index: 1;
`;

export default App;
