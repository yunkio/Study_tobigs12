from House import House
class Vile(House):
    def __init__(self,address,size_vile,rooms_vile,yard):
        self.address = address
        self.size_vile = size_vile
        self.rooms_vile = rooms_vile
        self.yard = yard

    def __str__(self):
        if(self.yard=='T'):
            return '{}에 위치해 있는 {}개의 방이 있고 마당이 있는 {}평의 주택입니다.'.format(self.address, self.rooms_vile, self.size_vile)
        if(self.yard=='F'):
            return '{}에 위치해 있는 {}개의 방이 있고 마당이 없는 {}평의 주택입니다.'.format(self.address, self.rooms_vile, self.size_vile)
