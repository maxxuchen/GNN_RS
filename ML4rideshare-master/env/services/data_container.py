"""
A container that temporally holds requests and rides information
"""


class Data_Container(object):
    def __init__(self):
        self.list = []

    def add(self, new_list):
        """
        reserve old list and add new items
        """
        for item in new_list:
            self.list.append(item)

    def delete(self, item):
        assert item in self.list
        self.list.remove(item)

    def renew(self, new_list):
        """
        delete old list
        """
        self.list = new_list

    def get_list(self):
        return self.list

    def reset(self):
        self.list = []

    def find_id(self, id):
        for item in self.list:
            if item.id == id:
                return item
        return None

    def check_none(self):
        if len(self.list) == 0:
            return True
        else:
            return False
