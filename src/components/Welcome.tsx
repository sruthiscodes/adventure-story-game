import React from 'react';
import styled from '@emotion/styled';
import { motion } from 'framer-motion';
import { useGameStore } from '../store/gameStore';

interface WelcomeProps {
  onNewGame: () => void;
  onLoadGame: () => void;
}

const Welcome: React.FC<WelcomeProps> = ({ onNewGame, onLoadGame }) => {
  const { loadGame, clearPersistedState } = useGameStore();
  
  const hasSavedGame = loadGame();
  
  const handleClearSave = () => {
    clearPersistedState();
    window.location.reload(); // Force a refresh to ensure state is cleared
  };
  
  return (
    <Container
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8 }}
    >
      <Title
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.3, duration: 0.5 }}
      >
        AI Storyteller
      </Title>
      
      <Subtitle
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.5, duration: 0.5 }}
      >
        An Interactive Adventure
      </Subtitle>
      
      <Description
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.7, duration: 0.5 }}
      >
        Embark on a unique journey where every choice matters. Your story adapts to your decisions, creating a personalized adventure that evolves with you.
      </Description>
      
      <ButtonContainer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.9, duration: 0.5 }}
      >
        <NewGameButton 
          onClick={onNewGame}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.98 }}
        >
          New Adventure
        </NewGameButton>
        
        {hasSavedGame && (
          <>
            <LoadGameButton 
              onClick={onLoadGame}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
            >
              Continue Adventure
            </LoadGameButton>
            
            <ClearSaveButton
              onClick={handleClearSave}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
            >
              Clear Saved Game
            </ClearSaveButton>
          </>
        )}
      </ButtonContainer>
    </Container>
  );
};

const Container = styled(motion.div)`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  padding: 2rem;
  text-align: center;
  position: relative;
  z-index: 1;
`;

const Title = styled(motion.h1)`
  font-size: 4rem;
  margin: 0;
  margin-bottom: 0.5rem;
  color: #ffd700;
  text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
`;

const Subtitle = styled(motion.h2)`
  font-size: 1.8rem;
  margin: 0;
  margin-bottom: 2rem;
  color: #f0f0f0;
  font-weight: normal;
`;

const Description = styled(motion.p)`
  max-width: 600px;
  margin-bottom: 3rem;
  color: #ccc;
  line-height: 1.6;
  font-size: 1.1rem;
`;

const ButtonContainer = styled(motion.div)`
  display: flex;
  gap: 1.5rem;
  
  @media (max-width: 480px) {
    flex-direction: column;
    width: 100%;
    max-width: 300px;
  }
`;

const NewGameButton = styled(motion.button)`
  padding: 1rem 2rem;
  background-color: #ffd700;
  color: #000;
  border: none;
  border-radius: 4px;
  font-size: 1.2rem;
  font-weight: bold;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: #ffec1f;
  }
`;

const LoadGameButton = styled(motion.button)`
  padding: 1rem 2rem;
  background-color: transparent;
  color: #ffd700;
  border: 2px solid #ffd700;
  border-radius: 4px;
  font-size: 1.2rem;
  font-weight: bold;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: rgba(255, 215, 0, 0.1);
  }
`;

const ClearSaveButton = styled(motion.button)`
  padding: 0.5rem 1rem;
  background-color: #333;
  color: #ff6b6b;
  border: 1px solid #ff6b6b;
  border-radius: 4px;
  font-size: 0.9rem;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: rgba(255, 107, 107, 0.1);
  }
`;

export default Welcome;