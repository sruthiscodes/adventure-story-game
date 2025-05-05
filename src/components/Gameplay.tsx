import React, { useState, useEffect } from 'react';
import styled from '@emotion/styled';
import { keyframes, css } from '@emotion/react';
import { motion, AnimatePresence } from 'framer-motion';
import { useGameStore } from '../store/gameStore';
import { Choice } from '../models/types';

// Add animation for stat changes
interface StatChangeAnimationProps {
  value: number;
  duration?: number;
}

// StatChangeAnimation component to show stat changes
const StatChangeAnimation: React.FC<StatChangeAnimationProps> = ({ value, duration = 1000 }) => {
  const [visible, setVisible] = useState(true);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(false);
    }, duration);
    
    return () => clearTimeout(timer);
  }, [duration]);
  
  if (!visible) return null;
  
  const prefix = value > 0 ? '+' : '';
  
  return (
    <AnimatedValue increase={value > 0}>
      {prefix}{value}
    </AnimatedValue>
  );
};

const Gameplay: React.FC = () => {
  const { 
    character, 
    currentStoryNode, 
    makeChoice, 
    saveGame,
    currentImage,
    setCurrentImage
  } = useGameStore();
  
  const [isChoosing, setIsChoosing] = useState(false);
  const [saveFeedback, setSaveFeedback] = useState('');
  const [imageLoading, setImageLoading] = useState(true);

  // Add new state to track stat changes
  const [statChanges, setStatChanges] = useState<{
    health: number | null;
    strength: number | null;
    intelligence: number | null;
    charisma: number | null;
  }>({
    health: null,
    strength: null,
    intelligence: null,
    charisma: null,
  });

  const handleSaveGame = () => {
    saveGame();
    setSaveFeedback('Game saved!');
    setTimeout(() => setSaveFeedback(''), 2000);
  };

  const handleChoiceClick = async (choice: Choice) => {
    setIsChoosing(true);
    setImageLoading(true);
    await makeChoice(choice);
    setIsChoosing(false);
  };
  
  // Effect to handle image loading
  useEffect(() => {
    if (currentImage) {
      const img = new Image();
      img.onload = () => {
        setImageLoading(false);
      };
      img.src = currentImage;
    }
  }, [currentImage]);

  return (
    <Container
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <GameHeader>
        <CharacterInfo>
          <CharacterName>{character.name}</CharacterName>
          <CharacterBackground>{character.background}</CharacterBackground>
        </CharacterInfo>
        <SaveButton onClick={handleSaveGame}>Save Game</SaveButton>
        {saveFeedback && <SaveFeedback>{saveFeedback}</SaveFeedback>}
      </GameHeader>

      <GameLayout>
        <StatsPanel>
          <StatsTitle>Character Stats</StatsTitle>
          <StatItem>
            <StatLabel>Health:</StatLabel>
            <StatValueContainer>
              <StatValue>{character.attributes.health}</StatValue>
              {statChanges.health !== null && (
                <StatChangeAnimation value={statChanges.health} />
              )}
            </StatValueContainer>
          </StatItem>
          <StatItem>
            <StatLabel>Strength:</StatLabel>
            <StatValueContainer>
              <StatValue>{character.attributes.strength}</StatValue>
              {statChanges.strength !== null && (
                <StatChangeAnimation value={statChanges.strength} />
              )}
            </StatValueContainer>
          </StatItem>
          <StatItem>
            <StatLabel>Intelligence:</StatLabel>
            <StatValueContainer>
              <StatValue>{character.attributes.intelligence}</StatValue>
              {statChanges.intelligence !== null && (
                <StatChangeAnimation value={statChanges.intelligence} />
              )}
            </StatValueContainer>
          </StatItem>
          <StatItem>
            <StatLabel>Charisma:</StatLabel>
            <StatValueContainer>
              <StatValue>{character.attributes.charisma}</StatValue>
              {statChanges.charisma !== null && (
                <StatChangeAnimation value={statChanges.charisma} />
              )}
            </StatValueContainer>
          </StatItem>
        </StatsPanel>

        <StoryPanel>
          <ImageContainer>
            {currentImage && !imageLoading ? (
              <StoryImage
                src={currentImage}
                alt="Story scene"
              />
            ) : (
              <ImageLoadingContainer>
                <LoadingAnimation style={{ width: '60px', height: '60px' }} />
              </ImageLoadingContainer>
            )}
          </ImageContainer>

          <StoryContent>
            <AnimatePresence mode="wait">
              <StoryText
                key={currentStoryNode.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.5 }}
              >
                {currentStoryNode.content}
              </StoryText>
            </AnimatePresence>
          </StoryContent>

          <ChoicesContainer>
            <AnimatePresence>
              {!isChoosing && currentStoryNode.choices.map((choice) => (
                <ChoiceButton
                  key={choice.id}
                  onClick={() => handleChoiceClick(choice)}
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.98 }}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.3 }}
                >
                  {choice.text}
                </ChoiceButton>
              ))}
              {isChoosing && (
                <LoadingChoices>
                  <LoadingText>Generating story...</LoadingText>
                  <LoadingAnimation />
                </LoadingChoices>
              )}
            </AnimatePresence>
          </ChoicesContainer>
        </StoryPanel>
      </GameLayout>
    </Container>
  );
};

const Container = styled(motion.div)`
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 1.5rem;
  position: relative;
  z-index: 1;
`;

const GameHeader = styled.header`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid rgba(255, 215, 0, 0.3);
  position: relative;
`;

const CharacterInfo = styled.div`
  display: flex;
  flex-direction: column;
`;

const CharacterName = styled.h2`
  margin: 0;
  color: #ffd700;
`;

const CharacterBackground = styled.p`
  margin: 0.25rem 0 0;
  color: #ccc;
  font-style: italic;
`;

const SaveButton = styled.button`
  padding: 0.5rem 1rem;
  background-color: #444;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
  &:hover {
    background-color: #555;
  }
`;

const SaveFeedback = styled.span`
  position: absolute;
  right: 0;
  bottom: -0.5rem;
  font-size: 0.8rem;
  color: #4caf50;
`;

const GameLayout = styled.div`
  display: grid;
  grid-template-columns: 250px 1fr;
  gap: 1.5rem;
  height: calc(100vh - 160px);
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    grid-template-rows: auto 1fr;
  }
`;

const StatsPanel = styled.div`
  background: rgba(0, 0, 0, 0.7);
  border-radius: 8px;
  padding: 1.5rem;
  width: 250px;
  height: fit-content;
`;

const StatsTitle = styled.h3`
  color: #ffd700;
  margin-top: 0;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid rgba(255, 215, 0, 0.3);
`;

const StatItem = styled.div`
  display: flex;
  justify-content: space-between;
  padding: 0.5rem 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
`;

const StatLabel = styled.span`
  color: #ccc;
`;

const StatValue = styled.span`
  color: #fff;
  font-weight: bold;
`;

const StoryPanel = styled.div`
  display: flex;
  flex-direction: column;
  background-color: rgba(0, 0, 0, 0.7);
  border-radius: 8px;
  overflow: auto;
  max-height: calc(100vh - 160px);
`;

const ImageContainer = styled.div`
  width: 100%;
  height: 0;
  padding-top: 60%; /* 16:9 Aspect Ratio */
  position: relative;
  border-radius: 8px 8px 0 0;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.3);
`;

const StoryImage = styled.img`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
`;

const StoryContent = styled.div`
  padding: 1.5rem;
  max-height: 300px;
  overflow-y: auto;
`;

const StoryText = styled(motion.p)`
  margin: 0;
  color: #f0f0f0;
  font-size: 1.1rem;
  line-height: 1.6;
  white-space: pre-wrap;
`;

const ChoicesContainer = styled.div`
  padding: 1rem;
  border-top: 1px solid rgba(255, 215, 0, 0.3);
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  min-height: 150px;
  max-height: 250px;
  overflow-y: auto;
`;

const ChoiceButton = styled(motion.button)`
  padding: 0.75rem 1.5rem;
  background-color: #333;
  color: #fff;
  border: 1px solid #555;
  border-radius: 4px;
  cursor: pointer;
  text-align: left;
  font-size: 1rem;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: #444;
    border-color: #666;
  }
`;

const LoadingChoices = styled.div`
  padding: 1rem;
  text-align: center;
  color: #ccc;
`;

const LoadingText = styled.p`
  margin: 0 0 1rem 0;
`;

const LoadingAnimation = styled.div`
  width: 50px;
  height: 50px;
  border: 3px solid rgba(255, 215, 0, 0.3);
  border-radius: 50%;
  border-top-color: #ffd700;
  animation: spin 1s linear infinite;
  margin: 0 auto;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const ImageLoadingContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.3);
`;

// Add new styled components for animations
const fadeAndSlideUp = keyframes`
  0% {
    opacity: 0;
    transform: translateY(0);
  }
  20% {
    opacity: 1;
  }
  80% {
    opacity: 1;
  }
  100% {
    opacity: 0;
    transform: translateY(-20px);
  }
`;

const AnimatedValue = styled.span<{ increase: boolean }>`
  position: absolute;
  font-weight: bold;
  font-size: 1rem;
  animation: ${fadeAndSlideUp} 1s ease-in-out forwards;
  color: ${props => props.increase ? '#4caf50' : '#f44336'};
  margin-left: 8px;
`;

const StatValueContainer = styled.div`
  display: flex;
  align-items: center;
  position: relative;
`;

export default Gameplay;