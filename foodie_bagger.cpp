#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <vector>
#include <string>

namespace py = pybind11;
using namespace std;

// -------------------- Structures --------------------

struct Item {
    string name;
    string size;
    string type;
    bool fragile;
    bool processed = false;
};

struct Bag {
    string type;
    vector<Item> items;
    bool hasFragile = false;
};

// -------------------- Globals --------------------

vector<Bag> bags;

// -------------------- Helper Functions --------------------

int getCapacity(const string& bagType) {
    if (bagType == "large") return 3;
    if (bagType == "medium") return 4;
    if (bagType == "small") return 5;
    if (bagType == "freezer") return 3;
    return 3;
}

bool isHeavy(const Item& item) {
    return item.size == "large";
}

bool bagFull(const Bag& bag) {
    return bag.items.size() >= getCapacity(bag.type);
}

bool canPlace(const Bag& bag, const Item& item) {

    if (bag.hasFragile && isHeavy(item)) {
        return false;
    }

    if (item.fragile) {
        for (const auto& i : bag.items) {
            if (isHeavy(i)) return false;
        }
    }

    return true;
}

Bag createBag(const string& type) {
    Bag b;
    b.type = type;

    cout << "Creating new " << type << " bag\n";

    return b;
}

void placeItem(Bag& bag, const Item& item) {

    bag.items.push_back(item);

    if (item.fragile)
        bag.hasFragile = true;

    cout << "Placed " << item.name
         << " in " << bag.type << " bag\n";
}

// -------------------- Core Logic --------------------

void processItems(vector<Item>& items, string step) {

    cout << "\n--- Step: " << step << " ---\n";

    for (auto& item : items) {

        // Rule R1:
        // Skip items already processed
        if (item.processed)
            continue;

        // Rule R2:
        // Skip items not belonging to this size step
        // unless they are frozen
        if (item.size != step &&
            item.type != "frozen")
            continue;

        // Rule R4:
        // Frozen items go to freezer bags
        if (item.type == "frozen") {

            bool placed = false;

            for (auto& bag : bags) {

                // Rule R5:
                // Place in existing freezer bag if possible
                if (bag.type == "freezer" &&
                    !bagFull(bag)) {

                    placeItem(bag, item);

                    placed = true;

                    break;
                }
            }

            // Rule R6:
            // Create freezer bag if none available
            if (!placed) {

                Bag newBag = createBag("freezer");

                placeItem(newBag, item);

                bags.push_back(newBag);
            }

            // Rule R7:
            // Mark frozen item as processed
            item.processed = true;

            continue;
        }

        // Rule R8:
        // Try placing normal items into existing bags
        bool placed = false;

        for (auto& bag : bags) {

            if (bag.type != step)
                continue;

            if (!bagFull(bag) &&
                canPlace(bag, item)) {

                placeItem(bag, item);

                placed = true;

                break;
            }
        }

        // Rule R9:
        // Create new bag if necessary
        if (!placed) {

            Bag newBag = createBag(step);

            placeItem(newBag, item);

            bags.push_back(newBag);
        }

        // Rule R10:
        // Mark item as processed
        item.processed = true;
    }
}


// -------------------- Python Interface --------------------

void clearBags() {
    bags.clear();
}

void processAllItems(vector<Item>& items) {

    clearBags();
    
    processItems(items, "large");
    processItems(items, "medium");
    processItems(items, "small");

}


vector<Bag> getBags() {
    return bags;
}

PYBIND11_MODULE(foodie_bagger, m) {

    m.doc() = "Grocery bagging system";

    // ---------------- Item ----------------

    py::class_<Item>(m, "Item")
        .def(py::init<>())
        .def_readwrite("name", &Item::name)
        .def_readwrite("size", &Item::size)
        .def_readwrite("type", &Item::type)
        .def_readwrite("fragile", &Item::fragile)
        .def_readwrite("processed", &Item::processed);

    // ---------------- Bag ----------------

    py::class_<Bag>(m, "Bag")
        .def(py::init<>())
        .def_readwrite("type", &Bag::type)
        .def_readwrite("items", &Bag::items)
        .def_readwrite("hasFragile", &Bag::hasFragile);

    // ---------------- Functions ----------------

    m.def("processItems", &processItems);


    m.def("clearBags", &clearBags);

    m.def("processAllItems", &processAllItems);


    m.def("getBags", &getBags);
}