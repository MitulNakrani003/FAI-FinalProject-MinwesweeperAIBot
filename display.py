import pygame


class DisplayGame():
    def __init__(self, state):
        self.CLOCK = None
        self.SCREEN = None
        self.font = None
        self.GRAY_SHADES = [
            (39, 43, 48),
            (34, 40, 49),
            (238, 238, 238),
        ]
        self.WHITE = (255, 255, 255)  
        self.YELLOW = [
                    (181, 172, 0),
                    (181,165,0),
                    (181,160,0),
                    (181,155,0),
                    (181,150,0)
                    ]
        h, w = state.shape
        self.blockSize = 80
        self.WINDOW_HEIGHT = h * self.blockSize
        self.WINDOW_WIDTH = w * self.blockSize
        self.state = state
        self.init()

    def init(self):
        pygame.init()
        self.font = pygame.font.SysFont('Courier', 18,bold=True)
        self.SCREEN = pygame.display.set_mode((self.WINDOW_HEIGHT, self.WINDOW_WIDTH))
        self.CLOCK = pygame.time.Clock()
        self.SCREEN.fill(self.GRAY_SHADES[1])
    
    def draw(self):
        self.drawGrid()
        pygame.display.update()
    
    def bugfix(self):
        return pygame.event.get()

    # def addText(self,no,x,y,color):
    #     self.SCREEN.blit(self.font.render(str(no), True, color), (x, y))
    #     pygame.display.update()

    def addText(self, no, x, y, color):
        text = str(no)
        text_surface = self.font.render(text, True, color)
        text_width, text_height = self.font.size(text)
        
        # Calculate the center position of the text
        center_x = x + (self.blockSize - text_width) / 2
        center_y = y + (self.blockSize - text_height) / 2
        
        # Blit the text at the calculated position
        self.SCREEN.blit(text_surface, (center_x, center_y))
        pygame.display.update()



    def drawGrid(self):
        j=0
        for column in range(0, self.WINDOW_WIDTH, self.blockSize):
            i=0
            for row in range(0, self.WINDOW_HEIGHT, self.blockSize):
                if(self.state[i][j]==-1):
                    pygame.draw.rect(self.SCREEN, self.GRAY_SHADES[0], [column,row,self.blockSize,self.blockSize])
                    self.addText("🟦", column + 10, row + 7, self.GRAY_SHADES[2])
                if(self.state[i][j]==0):
                    pygame.draw.rect(self.SCREEN, self.GRAY_SHADES[2], [column,row,self.blockSize,self.blockSize])
                    self.addText("🟦", column + 10, row + 7, self.GRAY_SHADES[2])
                elif(self.state[i][j]>0):
                    pygame.draw.rect(self.SCREEN, self.YELLOW[0], [column,row,self.blockSize,self.blockSize])
                    self.addText(self.state[i][j],column+10,row+7,self.GRAY_SHADES[2])
                i+=1
            j+=1
    


    